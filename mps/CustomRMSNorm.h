#pragma once

#include <torch/extension.h>
#include <utility>  // for std::pair

#ifdef __cplusplus
extern "C" {
#endif

// Constructor, destructor, and forward pass for our RMSNorm implementation
void* rms_norm_new(int dim, double eps, bool half_precision);
void rms_norm_free(void* ptr);

#ifdef __cplusplus
}
#endif
at::Tensor rms_norm_forward(void* ptr, at::Tensor& input, at::Tensor& weight);
std::pair<at::Tensor, at::Tensor> rms_norm_backward(void* ptr, at::Tensor& grad_output, at::Tensor& input, at::Tensor& weight);