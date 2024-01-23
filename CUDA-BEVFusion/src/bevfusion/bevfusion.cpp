/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "bevfusion.hpp"

#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"
#include "tensor.hpp"

namespace bevfusion {

static void save_half_device_tensor(const nvtype::half* data,
                                    std::vector<int32_t> shapes,
                                    std::string tensor_filename, void* stream) {
  nv::Tensor tensor = nv::Tensor::create(shapes, nv::DataType::Float16, true);
  tensor.copy_from_device(data, stream);
  // tensor.save(tensor_filename);
  int perline_num = shapes[shapes.size() - 1];
  int line_num = 5;
  //   for (int i = 0; i < shapes.size() - 1; i++) {
  //     line_num *= shapes[i];
  //   }
  tensor.print("Tensor", 0UL, perline_num, line_num);
  printf("==================[has save %s ]===================\n",
         tensor_filename.c_str());
}

class CoreImplement : public Core {
 public:
  virtual ~CoreImplement() {
    if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
    if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));
  }

  bool init(const CoreParameter& param) {
    camera_backbone_ = camera::create_backbone(param.camera_model);
    if (camera_backbone_ == nullptr) {
      printf("Failed to create camera backbone.\n");
      return false;
    }

    camera_bevpool_ = camera::create_bevpool(camera_backbone_->camera_shape(),
                                             param.geometry.geometry_dim.x,
                                             param.geometry.geometry_dim.y);
    if (camera_bevpool_ == nullptr) {
      printf("Failed to create camera bevpool.\n");
      return false;
    }

    camera_vtransform_ = camera::create_vtransform(param.camera_vtransform);
    if (camera_vtransform_ == nullptr) {
      printf("Failed to create camera vtransform.\n");
      return false;
    }

    transfusion_ = fuser::create_transfusion(param.transfusion);
    if (transfusion_ == nullptr) {
      printf("Failed to create transfusion.\n");
      return false;
    }

    transbbox_ = head::transbbox::create_transbbox(param.transbbox);
    if (transbbox_ == nullptr) {
      printf("Failed to create head transbbox.\n");
      return false;
    }

    lidar_scn_ = lidar::create_scn(param.lidar_scn);
    if (lidar_scn_ == nullptr) {
      printf("Failed to create lidar scn.\n");
      return false;
    }

    normalizer_ = camera::create_normalization(param.normalize);
    if (normalizer_ == nullptr) {
      printf("Failed to create normalizer.\n");
      return false;
    }

    camera_depth_ = camera::create_depth(param.normalize.output_width,
                                         param.normalize.output_height,
                                         param.normalize.num_camera);
    if (camera_depth_ == nullptr) {
      printf("Failed to create depth.\n");
      return false;
    }

    camera_geometry_ = camera::create_geometry(param.geometry);
    if (camera_geometry_ == nullptr) {
      printf("Failed to create geometry.\n");
      return false;
    }

    capacity_points_ = 300000;
    depth_stride_ = 6 * 704 * 256;
    bytes_capacity_points_ = capacity_points_ *
                             param.lidar_scn.voxelization.num_feature *
                             sizeof(nvtype::half);
    checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
    checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));
    param_ = param;
    return true;
  }

  std::vector<head::transbbox::BoundingBox> forward_only(
      const void* camera_images, const nvtype::half* lidar_points,
      int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf(
          "If it exceeds %d points, the default processing will simply crop it "
          "out.\n",
          cappoints);
    }

    num_points = std::min(cappoints, num_points);

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    size_t bytes_points = num_points *
                          param_.lidar_scn.voxelization.num_feature *
                          sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points,
                                 cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_,
                                 bytes_points, cudaMemcpyHostToDevice,
                                 _stream));

    const nvtype::half* lidar_feature =
        this->lidar_scn_->forward(lidar_points_device_, num_points, stream);
    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      normed_images = (nvtype::half*)this->normalizer_->forward(
          (const unsigned char**)(camera_images), stream);
    }
    const nvtype::half* depth = this->camera_depth_->forward(
        lidar_points_device_, num_points, 5, stream);

    this->camera_backbone_->forward(normed_images, depth, stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(),
        this->camera_geometry_->indices(), this->camera_geometry_->intervals(),
        this->camera_geometry_->num_intervals(), stream);

    const nvtype::half* camera_bevfeat =
        camera_vtransform_->forward(camera_bev, stream);
    const nvtype::half* fusion_feature =
        this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    return this->transbbox_->forward(fusion_feature,
                                     param_.transbbox.confidence_threshold,
                                     stream, param_.transbbox.sorted_bboxes);
  }

  std::vector<head::transbbox::BoundingBox> forward_timer(
      const void* camera_images, const nvtype::half* lidar_points,
      int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf(
          "If it exceeds %d points, the default processing will simply crop it "
          "out.\n",
          cappoints);
    }

    num_points = std::min(cappoints, num_points);

    printf("==================BEVFusion===================\n");
    std::vector<float> times;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    timer_.start(_stream);

    size_t bytes_points = num_points *
                          param_.lidar_scn.voxelization.num_feature *
                          sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points,
                                 cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_,
                                 bytes_points, cudaMemcpyHostToDevice,
                                 _stream));
    timer_.stop("[NoSt] CopyLidar");

    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      timer_.start(_stream);
      normed_images = (nvtype::half*)this->normalizer_->forward(
          (const unsigned char**)(camera_images), stream);
      timer_.stop("[NoSt] ImageNrom");
    }

    timer_.start(_stream);
    const nvtype::half* lidar_feature =
        this->lidar_scn_->forward(lidar_points_device_, num_points, stream);
    times.emplace_back(timer_.stop("Lidar Backbone"));

    timer_.start(_stream);
    const nvtype::half* depth = this->camera_depth_->forward(
        lidar_points_device_, num_points, 5, stream);
    times.emplace_back(timer_.stop("Camera Depth"));
    // unsigned int* depth_i = this->camera_depth_->depthCount();
    // for (int i = 0; i < 6 * 256 * 704; ++i) {
    //   if (*(depth_i + i) > 0) {
    //     printf("[DATE: %s] LINE: %d [%s] INFO %d\n", __DATE__, __LINE__,
    //            __func__, *(depth_i + i));
    //   }
    // }

    // printf("[DATE: %s] LINE: %d [%s] INFO %ld\n",__DATE__, __LINE__,
    // __func__, sizeof(depth->__x)); depth_count_ =
    // this->camera_depth_->depthCount();  // 6*704*256
    timer_.start(_stream);
    this->camera_backbone_->forward(normed_images, depth, stream);
    times.emplace_back(timer_.stop("Camera Backbone"));

    // 打印
    // std::vector<int> feature_shape = this->camera_backbone_->feature_shape();
    // for (auto dim : feature_shape) printf("feature_shape: %d\n", dim);
    timer_.start(_stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(),
        this->camera_geometry_->indices(), this->camera_geometry_->intervals(),
        this->camera_geometry_->num_intervals(), stream);
    times.emplace_back(timer_.stop("Camera Bevpool"));

    timer_.start(_stream);
    const nvtype::half* camera_bevfeat =
        camera_vtransform_->forward(camera_bev, stream);
    times.emplace_back(timer_.stop("VTransform"));

    timer_.start(_stream);
    const nvtype::half* fusion_feature =
        this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    times.emplace_back(timer_.stop("Transfusion"));

    // std::vector<int> camera_bevfeat_shape = {1, 80, 180, 180};
    // nv::Tensor camera_tensor = nv::Tensor::from_data_reference(
    //     (void*)camera_bevfeat, camera_bevfeat_shape, nv::DataType::UInt16);
    // std::string camera_path = "./camera.tensor";
    // nv::Tensor::save(camera_tensor, camera_path, stream);
    // std::vector<int> lidar_feature_shape = {1, 256, 180, 180};
    // nv::Tensor lidar_tensor = nv::Tensor::from_data_reference(
    //     (void*)lidar_feature, lidar_feature_shape, nv::DataType::UInt16);
    // std::string lidar_path = "./lidar.tensor";
    // nv::Tensor::save(lidar_tensor, lidar_path, stream);
    // std::vector<int> fusion_feature_shape = {1, 512, 180, 180};
    // nv::Tensor middle_tensor = nv::Tensor::from_data_reference(
    //     (void*)fusion_feature, fusion_feature_shape, nv::DataType::UInt16);
    // std::string middle_path = "./middle.tensor";
    // nv::Tensor::save(middle_tensor, middle_path, stream);
    // save_half_device_tensor(camera_bevfeat, camera_bevfeat_shape, "camera",
    // stream); save_half_device_tensor(lidar_feature, lidar_feature_shape,
    // "lidar", stream); save_half_device_tensor(fusion_feature,
    // fusion_feature_shape, "fusion_feature", stream);

    timer_.start(_stream);
    auto output = this->transbbox_->forward(
        fusion_feature, param_.transbbox.confidence_threshold, stream,
        param_.transbbox.sorted_bboxes);
    times.emplace_back(timer_.stop("Head BoundingBox"));

    float total_time =
        std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
    printf("Total: %.3f ms\n", total_time);
    printf("=============================================\n");
    return output;
  }

  virtual std::vector<head::transbbox::BoundingBox> forward(
      const unsigned char** camera_images, const nvtype::half* lidar_points,
      int num_points, void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_images, lidar_points, num_points,
                                 stream, true);
    } else {
      return this->forward_only(camera_images, lidar_points, num_points, stream,
                                true);
    }
  }

  virtual std::vector<head::transbbox::BoundingBox> forward_no_normalize(
      const nvtype::half* camera_normed_images_device,
      const nvtype::half* lidar_points, int num_points, void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_normed_images_device, lidar_points,
                                 num_points, stream, false);
    } else {
      return this->forward_only(camera_normed_images_device, lidar_points,
                                num_points, stream, false);
    }
  }

  virtual void set_timer(bool enable) override { enable_timer_ = enable; }

  virtual void print() override {
    camera_backbone_->print();
    camera_vtransform_->print();
    transfusion_->print();
    transbbox_->print();
  }

  virtual void update(const float* camera2lidar, const float* camera_intrinsics,
                      const float* lidar2image, const float* img_aug_matrix,
                      void* stream) override {
    camera_depth_->update(img_aug_matrix, lidar2image, stream);
    camera_geometry_->update(camera2lidar, camera_intrinsics, img_aug_matrix,
                             stream);
    // for (int i = 0; i < 96; ++i)
    //   printf("[DATE: %s] LINE: %d [%s] INFO %f\n", __DATE__, __LINE__, __func__,
    //          camera2lidar[i]);
  }

  virtual void free_excess_memory() override {
    camera_geometry_->free_excess_memory();
  }

 private:
  CoreParameter param_;
  nv::EventTimer timer_;
  nvtype::half* lidar_points_device_ = nullptr;
  nvtype::half* lidar_points_host_ = nullptr;
  size_t capacity_points_ = 0;
  size_t bytes_capacity_points_ = 0;
  int* depth_count_ = nullptr;
  size_t depth_stride_ = 0;

  std::shared_ptr<camera::Normalization> normalizer_;
  std::shared_ptr<camera::Backbone> camera_backbone_;
  std::shared_ptr<camera::BEVPool> camera_bevpool_;
  std::shared_ptr<camera::VTransform> camera_vtransform_;
  std::shared_ptr<camera::Depth> camera_depth_;
  std::shared_ptr<camera::Geometry> camera_geometry_;
  std::shared_ptr<lidar::SCN> lidar_scn_;
  std::shared_ptr<fuser::Transfusion> transfusion_;
  std::shared_ptr<head::transbbox::TransBBox> transbbox_;
  float confidence_threshold_ = 0;
  bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace bevfusion