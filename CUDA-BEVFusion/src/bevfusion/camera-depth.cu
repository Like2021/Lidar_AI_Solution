/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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

#include <cuda_fp16.h>
#include <thrust/sort.h>

#include "camera-depth.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"

namespace bevfusion {
namespace camera {

typedef struct {
  half x, y, z;
} half3;

static __forceinline__ __device__ float project(float4 T, float3 p) { return T.x * p.x + T.y * p.y + T.z * p.z + T.w; }

__forceinline__ __device__ static float clampf(float x, float lower, float upper) {
  return x < lower ? lower : (x > upper ? upper : x);
}

static __global__ void compute_depth_kernel(
    unsigned int num_points, const half* points, const float4* img_aug_matrix,
    const float4* lidar2image, unsigned int points_channel, int num_camera,
    unsigned int image_width, unsigned int image_height, half* depth_out) {
  int tid = cuda_linear_index;
  if (tid >= num_points) return;

  half3 point_half = *(const half3*)(&points[points_channel * tid]);
  float3 point = make_float3(point_half.x, point_half.y, point_half.z);
  for (int icamera = 0; icamera < num_camera; ++icamera) {
    float dist =
        clampf(project(lidar2image[4 * icamera + 2], point), 1e-5, 1e5);
    float3 projed =
        make_float3(project(lidar2image[4 * icamera + 0], point) / dist,
                    project(lidar2image[4 * icamera + 1], point) / dist, dist);  // 3D -> 2D 投影的坐标 最后一个维度可以忽略
    float x = project(img_aug_matrix[4 * icamera + 0], projed);
    float y = project(img_aug_matrix[4 * icamera + 1], projed);

    // Here you must use the float type to determine the range. For int(-0.5),
    // its value is 0.
    if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
      int ix = static_cast<int>(x);
      int iy = static_cast<int>(y);
      depth_out[(icamera * image_height + iy) * image_width + ix] =
          __float2half(dist);
    }
  }
}

// static __global__ void compute_depth_kernel_int_version(unsigned int num_points, const half* points, const float4* img_aug_matrix,
//                                                         const float4* lidar2image, unsigned int points_channel, int num_camera,
//                                                         unsigned int image_width, unsigned int image_height, int* depth_out_i) {
//   int tid = cuda_linear_index;
//   if (tid >= num_points) return;

//   half3 point_half = *(const half3*)(&points[points_channel * tid]);
//   float3 point = make_float3(point_half.x, point_half.y, point_half.z);
//   for (int icamera = 0; icamera < num_camera; ++icamera) {
//     float dist = clampf(project(lidar2image[4 * icamera + 2], point), 1e-5, 1e5);
//     float3 projed = make_float3(project(lidar2image[4 * icamera + 0], point) / dist,
//                                 project(lidar2image[4 * icamera + 1], point) / dist, dist);
//     float x = project(img_aug_matrix[4 * icamera + 0], projed);
//     float y = project(img_aug_matrix[4 * icamera + 1], projed);

//     // Here you must use the float type to determine the range. For int(-0.5), its value is 0.
//     if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
//       int ix = static_cast<int>(x);
//       int iy = static_cast<int>(y);
//       atomicMin(&(depth_out_i[(icamera * image_height + iy) * image_width + ix]), __float_as_int(dist));
//     }
//   }
// }

// static __global__ void compute_depth_kernel_int2half(unsigned int num_pixels, const int* depth_out_i, half *depth_out) {
//   int tid = cuda_linear_index;
//   if (tid >= num_pixels) return;
//   if (__int_as_float(depth_out_i[tid]) < 200) {
//     depth_out[tid] =  __float2half(__int_as_float(depth_out_i[tid]));
//   }
// }

// static __device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
//   float old;
//   old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
//         __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

//   return old;
// }

// static __global__ void compute_depth_kernel_float_version(unsigned int num_points, const half* points, const float4* img_aug_matrix,
//                                                           const float4* lidar2image, unsigned int points_channel, int num_camera,
//                                                           unsigned int image_width, unsigned int image_height, float* depth_out_i) {
//   int tid = cuda_linear_index;
//   if (tid >= num_points) return;

//   half3 point_half = *(const half3*)(&points[points_channel * tid]);
//   float3 point = make_float3(point_half.x, point_half.y, point_half.z);
//   for (int icamera = 0; icamera < num_camera; ++icamera) {
//     float dist = clampf(project(lidar2image[4 * icamera + 2], point), 1e-5, 1e5);
//     float3 projed = make_float3(project(lidar2image[4 * icamera + 0], point) / dist,
//                                 project(lidar2image[4 * icamera + 1], point) / dist, dist);
//     float x = project(img_aug_matrix[4 * icamera + 0], projed);
//     float y = project(img_aug_matrix[4 * icamera + 1], projed);

//     // Here you must use the float type to determine the range. For int(-0.5), its value is 0.
//     if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
//       int ix = static_cast<int>(x);
//       int iy = static_cast<int>(y);
//       depth_out_i[(icamera * image_height + iy) * image_width + ix] = dist;
//     }
//   }
// }

// static __global__ void compute_depth_kernel_int_version(unsigned int num_points, const half* points, const float4* img_aug_matrix,
//                                                         const float4* lidar2image, unsigned int points_channel, int num_camera,
//                                                         unsigned int image_width, unsigned int image_height, int* depth_out_i) {
//   int tid = cuda_linear_index;
//   if (tid >= num_points) return;

//   half3 point_half = *(const half3*)(&points[points_channel * tid]);
//   float3 point = make_float3(point_half.x, point_half.y, point_half.z);
//   for (int icamera = 0; icamera < num_camera; ++icamera) {
//     float dist = clampf(project(lidar2image[4 * icamera + 2], point), 1e-5, 1e5);
//     float3 projed = make_float3(project(lidar2image[4 * icamera + 0], point) / dist,
//                                 project(lidar2image[4 * icamera + 1], point) / dist, dist);
//     float x = project(img_aug_matrix[4 * icamera + 0], projed);
//     float y = project(img_aug_matrix[4 * icamera + 1], projed);

//     // Here you must use the float type to determine the range. For int(-0.5), its value is 0.
//     if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
//       int ix = static_cast<int>(x);
//       int iy = static_cast<int>(y);
//       atomicMin(&(depth_out_i[(icamera * image_height + iy) * image_width + ix]), __float_as_int(dist));
//       // printf("__float_as_int(dist): %d\n", __float_as_int(dist));
//     }
//   }
// }

// static __global__ void compute_depth_kernel_int2float(unsigned int num_pixels, const int* depth_out_i, float *depth_out_f) {
//   int tid = cuda_linear_index;
//   if (tid >= num_pixels) return;
//   if (__int_as_float(depth_out_i[tid]) < 200) {
//     depth_out_f[tid] =  __int_as_float(depth_out_i[tid]);
//   }
// }

class DepthImplement : public Depth {
 public:
  virtual ~DepthImplement() {
    if (img_aug_matrix_) checkRuntime(cudaFree(img_aug_matrix_));
    if (lidar2image_) checkRuntime(cudaFree(lidar2image_));
    if (depth_output_) checkRuntime(cudaFree(depth_output_));
    if (depth_output_int_) checkRuntime(cudaFree(depth_output_int_));
    // if (depth_output_int_host_) checkRuntime(cudaFreeHost(depth_output_int_host_));
    // if (depth_output_float_) checkRuntime(cudaFree(depth_output_float_));
    // if (depth_output_float_host_) checkRuntime(cudaFreeHost(depth_output_float_host_));
  }

  bool init(unsigned int image_width, unsigned int image_height, unsigned int num_camera) {
    this->image_width_ = image_width;     // 704
    this->image_height_ = image_height;   // 256
    this->num_camera_ = num_camera;       // 6
    // this->source_height_ = 900;
    // this->source_width_ = 1600;

    bytes_of_matrix_ = num_camera_ * 4 * 4 * sizeof(float);
    checkRuntime(cudaMalloc(&img_aug_matrix_, bytes_of_matrix_));
    checkRuntime(cudaMalloc(&lidar2image_, bytes_of_matrix_));
    bytes_of_depth_ = num_camera_ * image_width_ * image_height_ * sizeof(half);
    checkRuntime(cudaMalloc(&depth_output_, bytes_of_depth_));
    bytes_of_depth_int_ = num_camera_ * image_width_ * image_height_ * sizeof(int);
    checkRuntime(cudaMalloc(&depth_output_int_, bytes_of_depth_int_));
    // checkRuntime(cudaMallocHost(&depth_output_int_host_, bytes_of_depth_int_));
    // bytes_of_depth_float_ = num_camera_ * image_width_ * image_height_ * sizeof(float);
    // checkRuntime(cudaMalloc(&depth_output_float_, bytes_of_depth_float_));
    // checkRuntime(cudaMallocHost(&depth_output_float_host_, bytes_of_depth_float_));
    return true;
  }

  // points must be of half-float type
  virtual nvtype::half* forward(const nvtype::half* points, int num_points, int points_dim, void* stream) override {
    Asserts(num_camera_ > 0, "Please initialize the parameters before forward");
    unsigned int num_pixels = num_camera_ * image_width_ * image_height_;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    checkRuntime(cudaMemsetAsync(depth_output_, 0, bytes_of_depth_, _stream));
    checkRuntime(cudaMemsetAsync(depth_output_int_, 127, bytes_of_depth_int_, _stream));
    cuda_linear_launch(compute_depth_kernel, _stream, num_points, reinterpret_cast<const half*>(points),
                       reinterpret_cast<const float4*>(img_aug_matrix_), reinterpret_cast<const float4*>(lidar2image_),
                       points_dim, num_camera_, image_width_, image_height_, depth_output_);

    // // 先转换成int
    // cuda_linear_launch(compute_depth_kernel_int_version, _stream, num_points, reinterpret_cast<const half*>(points),
    //                    reinterpret_cast<const float4*>(img_aug_matrix_), reinterpret_cast<const float4*>(lidar2image_),
    //                    points_dim, num_camera_, image_width_, image_height_, depth_output_int_);
    // // 最后变成half
    // cuda_linear_launch(compute_depth_kernel_int2half, _stream, num_pixels, depth_output_int_, depth_output_);

    
    // 
    // cuda_linear_launch(compute_depth_kernel_int_version, _stream, num_points, reinterpret_cast<const half*>(points),
    //                    reinterpret_cast<const float4*>(img_aug_matrix_), reinterpret_cast<const float4*>(lidar2image_),
    //                    points_dim, num_camera_, image_width_, image_height_, depth_output_int_);
    // // // 再转换成float
    // cuda_linear_launch(compute_depth_kernel_int2float, _stream, num_pixels, depth_output_int_, depth_output_float_);

    // cuda_linear_launch(compute_depth_kernel_int2half, _stream, num_pixels, depth_output_int_, depth_output_);
    // checkRuntime(cudaMemcpyAsync(depth_output_float_host_, depth_output_float_, bytes_of_depth_float_, cudaMemcpyDeviceToHost, _stream)); 
    // checkRuntime(cudaMemcpyAsync(depth_output_int_host_, depth_output_int_, bytes_of_depth_int_, cudaMemcpyDeviceToHost, _stream)); 
    // for (int i = 0; i < 6 * 256 * 704; ++i) if (*(depth_output_float_host_ + i) > 5) printf("depth_output_int_host_: %f\n", (*(depth_output_float_host_ + i)));
    return (nvtype::half*)depth_output_;
  }

  // You can call this function if you need to update the matrix
  // All matrix pointers must be on the host
  virtual void update(const float* img_aug_matrix, const float* lidar2image, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    // For users, please ensure that the pointer lifecycle is available for asynchronous copying.
    checkRuntime(cudaMemcpyAsync(img_aug_matrix_, img_aug_matrix, bytes_of_matrix_, cudaMemcpyHostToDevice, _stream));
    checkRuntime(cudaMemcpyAsync(lidar2image_, lidar2image, bytes_of_matrix_, cudaMemcpyHostToDevice, _stream));
  }

  // virtual unsigned int* depthCount() override { return depth_output_int_host_; }

 private:
  size_t bytes_of_matrix_ = 0;
  float* img_aug_matrix_ = nullptr;
  float* lidar2image_ = nullptr;
  half* depth_output_ = nullptr;
  int* depth_output_int_ = nullptr;
  int* depth_output_int_host_ = nullptr;
  // float* depth_output_float_ = nullptr;
  // float* depth_output_float_host_ = nullptr;
  size_t bytes_of_depth_int_ = 0;
  // size_t bytes_of_depth_float_ = 0;
  size_t bytes_of_depth_ = 0;
  unsigned int image_height_ = 0;
  unsigned int image_width_ = 0;
  unsigned int num_camera_ = 0;
  // unsigned int source_height_ = 0;
  // unsigned int source_width_ = 0;
};

std::shared_ptr<Depth> create_depth(unsigned int image_width, unsigned int image_height, unsigned int num_camera) {
  std::shared_ptr<DepthImplement> instance(new DepthImplement());
  if (!instance->init(image_width, image_height, num_camera)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion