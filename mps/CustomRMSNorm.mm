#import <Metal/Metal.h>
#include <torch/extension.h>
#include "CustomRMSNorm.h"
#include <ATen/mps/MPSStream.h> // Public API for MPS streams
#include <pybind11/pybind11.h>
#include <iostream>
#include <algorithm>

// We no longer need the private/native declarations


// The Metal shader kernel for RMSNorm, written as a C++ raw string literal.
static const char* RMSNORM_KERNEL_FLOAT = R"(
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_kernel(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& threads_per_row [[buffer(6)]],
    constant uint& rows_per_tg [[buffer(7)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint global_row = tgid.y * rows_per_tg + tid.y;
    if (global_row >= num_rows) return;
    uint offset = global_row * dim;

    float local_sum_sq = 0.0f;
    bool vec4 = ((dim & 3u) == 0u) && ((threads_per_row & 3u) == 0u);
    if (vec4) {
        if ((tid.x & 3u) == 0u) {
            for (uint col = tid.x; col < dim; col += threads_per_row) {
                uint base = offset + col;
                float4 v = float4(in[base+0], in[base+1], in[base+2], in[base+3]);
                local_sum_sq += dot(v, v);
            }
        }
    } else {
        for (uint col = tid.x; col < dim; col += threads_per_row) {
            float val = in[offset + col];
            local_sum_sq += val * val;
        }
    }

    uint local_lane = tid.x % 32u;
    uint local_sgid = tid.x / 32u;
    local_sum_sq = simd_sum(local_sum_sq);

    uint num_local_sgs = (threads_per_row + 31u) / 32u;
    uint shared_offset = tid.y * num_local_sgs;
    if (local_lane == 0) {
        shared_mem[shared_offset + local_sgid] = local_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_sgid == 0) {
        float sg_sum = (local_lane < num_local_sgs) ? shared_mem[shared_offset + local_lane] : 0.0f;
        sg_sum = simd_sum(sg_sum);
        if (local_lane == 0) {
            shared_mem[shared_offset] = sg_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = shared_mem[shared_offset];
    float mean_sq = total_sum_sq / float(dim);
    float norm_factor = rsqrt(mean_sq + eps);

    if (vec4) {
        if ((tid.x & 3u) == 0u) {
            for (uint col = tid.x; col < dim; col += threads_per_row) {
                uint base = offset + col;
                float4 v = float4(in[base+0], in[base+1], in[base+2], in[base+3]);
                float4 ww = float4(weight[base - offset + 0], weight[base - offset + 1], weight[base - offset + 2], weight[base - offset + 3]);
                float4 r = v * norm_factor * ww;
                out[base+0] = r.x;
                out[base+1] = r.y;
                out[base+2] = r.z;
                out[base+3] = r.w;
            }
        }
    } else {
        for (uint col = tid.x; col < dim; col += threads_per_row) {
            float val = in[offset + col];
            out[offset + col] = val * norm_factor * weight[col];
        }
    }
}
)";

static const char* RMSNORM_KERNEL_HALF = R"(
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_kernel(
    device const half* in [[buffer(0)]],
    device half* out [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& threads_per_row [[buffer(6)]],
    constant uint& rows_per_tg [[buffer(7)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint global_row = tgid.y * rows_per_tg + tid.y;
    if (global_row >= num_rows) return;
    uint offset = global_row * dim;

    float local_sum_sq = 0.0f;
    bool vec4 = ((dim & 3u) == 0u) && ((threads_per_row & 3u) == 0u);
    if (vec4) {
        if ((tid.x & 3u) == 0u) {
            for (uint col = tid.x; col < dim; col += threads_per_row) {
                uint base = offset + col;
                float4 v = float4(half(in[base+0]), half(in[base+1]), half(in[base+2]), half(in[base+3]));
                local_sum_sq += dot(v, v);
            }
        }
    } else {
        for (uint col = tid.x; col < dim; col += threads_per_row) {
            half val = in[offset + col];
            local_sum_sq += float(val) * float(val);
        }
    }

    uint local_lane = tid.x % 32u;
    uint local_sgid = tid.x / 32u;
    local_sum_sq = simd_sum(local_sum_sq);

    uint num_local_sgs = (threads_per_row + 31u) / 32u;
    uint shared_offset = tid.y * num_local_sgs;
    if (local_lane == 0) {
        shared_mem[shared_offset + local_sgid] = local_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_sgid == 0) {
        float sg_sum = (local_lane < num_local_sgs) ? shared_mem[shared_offset + local_lane] : 0.0f;
        sg_sum = simd_sum(sg_sum);
        if (local_lane == 0) {
            shared_mem[shared_offset] = sg_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = shared_mem[shared_offset];
    float mean_sq = total_sum_sq / float(dim);
    float inv_rms_f = rsqrt(mean_sq + eps);
    half inv_rms = half(inv_rms_f);

    if (vec4) {
        if ((tid.x & 3u) == 0u) {
            for (uint col = tid.x; col < dim; col += threads_per_row) {
                uint base = offset + col;
                float4 v = float4(half(in[base+0]), half(in[base+1]), half(in[base+2]), half(in[base+3]));
                float4 ww = float4(half(weight[base - offset + 0]), half(weight[base - offset + 1]), half(weight[base - offset + 2]), half(weight[base - offset + 3]));
                float4 r = v * inv_rms_f * ww;
                out[base+0] = half(r.x);
                out[base+1] = half(r.y);
                out[base+2] = half(r.z);
                out[base+3] = half(r.w);
            }
        }
    } else {
        for (uint col = tid.x; col < dim; col += threads_per_row) {
            half val = in[offset + col];
            out[offset + col] = val * inv_rms * weight[col];
        }
    }
}
)";

static const char* RMSNORM_BACKWARD_KERNEL_FLOAT = R"(
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_backward_kernel(
    device const float* dy [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* dx [[buffer(2)]],
    device float* contrib [[buffer(3)]],
    device const float* weight [[buffer(4)]],
    constant uint& num_rows [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant uint& threads_per_row [[buffer(8)]],
    constant uint& rows_per_tg [[buffer(9)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint global_row = tgid.y * rows_per_tg + tid.y;
    if (global_row >= num_rows) return;
    uint offset = global_row * dim;

    float local_sum_sq = 0.0f;
    for (uint col = tid.x; col < dim; col += threads_per_row) {
        float val = x[offset + col];
        local_sum_sq += val * val;
    }

    uint local_lane = tid.x % 32u;
    uint local_sgid = tid.x / 32u;
    local_sum_sq = simd_sum(local_sum_sq);

    uint num_local_sgs = (threads_per_row + 31u) / 32u;
    uint shared_offset = tid.y * num_local_sgs;
    if (local_lane == 0) {
        shared_mem[shared_offset + local_sgid] = local_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_sgid == 0) {
        float sg_sum = (local_lane < num_local_sgs) ? shared_mem[shared_offset + local_lane] : 0.0f;
        sg_sum = simd_sum(sg_sum);
        if (local_lane == 0) {
            shared_mem[shared_offset] = sg_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = shared_mem[shared_offset];
    float mean_sq = total_sum_sq / float(dim);
    float inv_rms = rsqrt(mean_sq + eps);

    float local_sum1 = 0.0f;
    for (uint col = tid.x; col < dim; col += threads_per_row) {
        uint col_idx = offset + col;
        local_sum1 += weight[col] * dy[col_idx] * x[col_idx];
    }

    local_sum1 = simd_sum(local_sum1);
    if (local_lane == 0) {
        shared_mem[shared_offset + local_sgid] = local_sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_sgid == 0) {
        float sg_sum = (local_lane < num_local_sgs) ? shared_mem[shared_offset + local_lane] : 0.0f;
        sg_sum = simd_sum(sg_sum);
        if (local_lane == 0) {
            shared_mem[shared_offset] = sg_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum1 = shared_mem[shared_offset];
    float adjust = (total_sum1 / float(dim)) * (inv_rms * inv_rms * inv_rms);

    for (uint col = tid.x; col < dim; col += threads_per_row) {
        uint col_idx = offset + col;
        float norm_x = x[col_idx] * inv_rms;
        float dx_val = (weight[col] * dy[col_idx] * inv_rms) - (x[col_idx] * adjust);
        dx[col_idx] = dx_val;
        contrib[col_idx] = dy[col_idx] * norm_x;
    }
}
)";

static const char* RMSNORM_BACKWARD_KERNEL_HALF = R"(
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_backward_kernel(
    device const half* dy [[buffer(0)]],
    device const half* x [[buffer(1)]],
    device half* dx [[buffer(2)]],
    device half* contrib [[buffer(3)]],
    device const half* weight [[buffer(4)]],
    constant uint& num_rows [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant uint& threads_per_row [[buffer(8)]],
    constant uint& rows_per_tg [[buffer(9)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint global_row = tgid.y * rows_per_tg + tid.y;
    if (global_row >= num_rows) return;
    uint offset = global_row * dim;

    float local_sum_sq = 0.0f;
    for (uint col = tid.x; col < dim; col += threads_per_row) {
        half val = x[offset + col];
        local_sum_sq += float(val) * float(val);
    }

    uint local_lane = tid.x % 32u;
    uint local_sgid = tid.x / 32u;
    local_sum_sq = simd_sum(local_sum_sq);

    uint num_local_sgs = (threads_per_row + 31u) / 32u;
    uint shared_offset = tid.y * num_local_sgs;
    if (local_lane == 0) {
        shared_mem[shared_offset + local_sgid] = local_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_sgid == 0) {
        float sg_sum = (local_lane < num_local_sgs) ? shared_mem[shared_offset + local_lane] : 0.0f;
        sg_sum = simd_sum(sg_sum);
        if (local_lane == 0) {
            shared_mem[shared_offset] = sg_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = shared_mem[shared_offset];
    float mean_sq = total_sum_sq / float(dim);
    float inv_rms_f = rsqrt(mean_sq + eps);
    half inv_rms = half(inv_rms_f);

    float local_sum1 = 0.0f;
    for (uint col = tid.x; col < dim; col += threads_per_row) {
        uint col_idx = offset + col;
        local_sum1 += float(weight[col]) * float(dy[col_idx]) * float(x[col_idx]);
    }

    local_sum1 = simd_sum(local_sum1);
    if (local_lane == 0) {
        shared_mem[shared_offset + local_sgid] = local_sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_sgid == 0) {
        float sg_sum = (local_lane < num_local_sgs) ? shared_mem[shared_offset + local_lane] : 0.0f;
        sg_sum = simd_sum(sg_sum);
        if (local_lane == 0) {
            shared_mem[shared_offset] = sg_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum1 = shared_mem[shared_offset];
    float adjust_f = (total_sum1 / float(dim)) * (inv_rms_f * inv_rms_f * inv_rms_f);
    half adjust = half(adjust_f);

    for (uint col = tid.x; col < dim; col += threads_per_row) {
        uint col_idx = offset + col;
        half norm_x = x[col_idx] * inv_rms;
        half dx_val = (weight[col] * dy[col_idx] * inv_rms) - (x[col_idx] * adjust);
        dx[col_idx] = dx_val;
        contrib[col_idx] = dy[col_idx] * norm_x;
    }
}
)";

// Objective-C class to hold the pipeline state and parameters.
@interface CustomRMSNormImpl : NSObject
@property (nonatomic, strong) id<MTLComputePipelineState> pipelineState;
@property (nonatomic, strong) id<MTLComputePipelineState> pipelineStateBackward;
@property (nonatomic, strong) id<MTLComputePipelineState> pipelineStateReduce;
@property (nonatomic) int dim;
@property (nonatomic) float eps;
@property (nonatomic) bool half_precision;
@end

@implementation CustomRMSNormImpl

- (instancetype)initWithDevice:(id<MTLDevice>)device dim:(int)dim eps:(float)eps half:(bool)half {
    self = [super init];
    if (self) {
        self.dim = dim;
        self.eps = eps;
        self.half_precision = half;

        NSError* error = nil;
        NSString* fwdSource = half ? [NSString stringWithUTF8String:RMSNORM_KERNEL_HALF] : [NSString stringWithUTF8String:RMSNORM_KERNEL_FLOAT];
        NSString* bwdSource = half ? [NSString stringWithUTF8String:RMSNORM_BACKWARD_KERNEL_HALF] : [NSString stringWithUTF8String:RMSNORM_BACKWARD_KERNEL_FLOAT];
        // Simple row-reduction kernels (implemented below)
        NSString* redSource = half ? @"#include <metal_stdlib>\nusing namespace metal;\n\nkernel void rms_norm_reduce_kernel( device const half* contrib [[buffer(0)]], device float* dw [[buffer(1)]], constant uint& num_rows [[buffer(2)]], constant uint& dim [[buffer(3)]], uint gid [[thread_position_in_grid]] ) { if (gid >= dim) return; float sumv = 0.0f; for (uint r=0; r<num_rows; ++r) { sumv += float(contrib[r*dim + gid]); } dw[gid] = sumv; }" : @"#include <metal_stdlib>\nusing namespace metal;\n\nkernel void rms_norm_reduce_kernel( device const float* contrib [[buffer(0)]], device float* dw [[buffer(1)]], constant uint& num_rows [[buffer(2)]], constant uint& dim [[buffer(3)]], uint gid [[thread_position_in_grid]] ) { if (gid >= dim) return; float sumv = 0.0f; for (uint r=0; r<num_rows; ++r) { sumv += contrib[r*dim + gid]; } dw[gid] = sumv; }";

        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        options.mathMode = MTLMathModeFast;
#else
        options.fastMathEnabled = YES;
#endif

        id<MTLLibrary> libraryFwd = [device newLibraryWithSource:fwdSource options:options error:&error];
        if (!libraryFwd) {
            throw std::runtime_error("Failed to create Metal fwd library: " + std::string([[error description] UTF8String]));
        }
        id<MTLFunction> functionFwd = [libraryFwd newFunctionWithName:@"rms_norm_kernel"];
        if (!functionFwd) {
            throw std::runtime_error("Failed to find Metal fwd kernel function.");
        }
        self.pipelineState = [device newComputePipelineStateWithFunction:functionFwd error:&error];
        if (!self.pipelineState) {
            throw std::runtime_error("Failed to create fwd pipeline state: " + std::string([[error description] UTF8String]));
        }

        id<MTLLibrary> libraryBwd = [device newLibraryWithSource:bwdSource options:options error:&error];
        if (!libraryBwd) {
            throw std::runtime_error("Failed to create Metal bwd library: " + std::string([[error description] UTF8String]));
        }
        id<MTLFunction> functionBwd = [libraryBwd newFunctionWithName:@"rms_norm_backward_kernel"];
        if (!functionBwd) {
            throw std::runtime_error("Failed to find Metal bwd kernel function.");
        }
        self.pipelineStateBackward = [device newComputePipelineStateWithFunction:functionBwd error:&error];
        if (!self.pipelineStateBackward) {
            throw std::runtime_error("Failed to create bwd pipeline state: " + std::string([[error description] UTF8String]));
        }

        id<MTLLibrary> libraryRed = [device newLibraryWithSource:redSource options:options error:&error];
        if (!libraryRed) {
            throw std::runtime_error("Failed to create Metal reduce library: " + std::string([[error description] UTF8String]));
        }
        id<MTLFunction> functionRed = [libraryRed newFunctionWithName:@"rms_norm_reduce_kernel"];
        if (!functionRed) {
            throw std::runtime_error("Failed to find Metal reduce kernel function.");
        }
        self.pipelineStateReduce = [device newComputePipelineStateWithFunction:functionRed error:&error];
        if (!self.pipelineStateReduce) {
            throw std::runtime_error("Failed to create reduce pipeline state: " + std::string([[error description] UTF8String]));
        }
    }
    return self;
}

- (at::Tensor)forward:(at::Tensor&)input weight:(at::Tensor&)weight {
    // The input tensor is expected to be 2D: (num_rows, dim)
    id<MTLDevice> device = self.pipelineState.device;
    TORCH_CHECK(input.is_mps(), "Input tensor must be on MPS device");
    TORCH_CHECK(weight.is_mps(), "Weight tensor must be on MPS device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(input.size(1) == self.dim, "Input tensor's second dimension must match 'dim'");

    // Get the current MPS stream and command encoder using the public API
    at::mps::MPSStream* mpsStream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = mpsStream->commandEncoder();
    [encoder setComputePipelineState:self.pipelineState];

    at::Tensor input_cast = self.half_precision ? input.to(torch::kHalf) : input.to(torch::kFloat);
    at::Tensor weight_cast = self.half_precision ? weight.to(torch::kHalf) : weight.to(torch::kFloat);
    at::Tensor output = at::empty_like(input_cast);

    // Set buffers using input_cast and weight_cast
    [encoder setBuffer:(__bridge id<MTLBuffer>)input_cast.storage().data_ptr().get() offset:input_cast.storage_offset() atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output.storage().data_ptr().get() offset:output.storage_offset() atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight_cast.storage().data_ptr().get() offset:weight_cast.storage_offset() atIndex:2];
    int num_rows_int = (int)input.size(0);
    uint num_rows = static_cast<uint>(num_rows_int);
    uint u_dim = static_cast<uint>(self.dim);
    [encoder setBytes:&num_rows length:sizeof(uint) atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(uint) atIndex:4];
    float eps_f = self.eps;
    [encoder setBytes:&eps_f length:sizeof(float) atIndex:5];

    NSUInteger max_threads = self.pipelineState.maxTotalThreadsPerThreadgroup;
    NSUInteger min_threads_per_row = 32;
    NSUInteger threads_per_row = std::max(min_threads_per_row, (((NSUInteger)self.dim + 31) / 32 * 32));
    if (threads_per_row > max_threads) threads_per_row = max_threads;
    NSUInteger max_rows_per_tg = max_threads / threads_per_row;
    NSUInteger rows_per_tg = std::min(max_rows_per_tg, (NSUInteger)num_rows);
    if (rows_per_tg == 0) {
        rows_per_tg = 1;
        threads_per_row = max_threads;
    }
    NSUInteger grid_y = (num_rows + rows_per_tg - 1) / rows_per_tg;

    uint u_threads_per_row = static_cast<uint>(threads_per_row);
    uint u_rows_per_tg = static_cast<uint>(rows_per_tg);
    [encoder setBytes:&u_threads_per_row length:sizeof(uint) atIndex:6];
    [encoder setBytes:&u_rows_per_tg length:sizeof(uint) atIndex:7];

    // Allocate threadgroup memory for shared sum (always float-sized in shaders)
    size_t type_size = 4;
    NSUInteger num_sgs = (threads_per_row + 31) / 32;
    [encoder setThreadgroupMemoryLength:type_size * rows_per_tg * num_sgs atIndex:0];

    // Dispatch: Grid in y for threadgroup batches
    MTLSize threadgroup_size = MTLSizeMake(threads_per_row, rows_per_tg, 1);
    MTLSize grid_size = MTLSizeMake(1, grid_y, 1);

    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];

    // No need for endEncoding; PyTorch handles it
    // [encoder endEncoding];
    // PyTorch will handle committing the command buffer. We do not need to call commit.

    return output;
}

- (std::pair<at::Tensor, at::Tensor>)backward:(at::Tensor&)dy input:(at::Tensor&)input weight:(at::Tensor&)weight {
    id<MTLDevice> device = self.pipelineStateBackward.device;
    TORCH_CHECK(dy.is_mps(), "Grad output must be on MPS device");
    TORCH_CHECK(input.is_mps(), "Input tensor must be on MPS device");
    TORCH_CHECK(weight.is_mps(), "Weight tensor must be on MPS device");
    TORCH_CHECK(dy.is_contiguous(), "Grad output must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(input.size(1) == self.dim, "Input tensor's second dimension must match 'dim'");

    at::mps::MPSStream* mpsStream = at::mps::getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = mpsStream->commandEncoder();
    [encoder setComputePipelineState:self.pipelineStateBackward];

    at::Tensor dy_cast = self.half_precision ? dy.to(torch::kHalf) : dy.to(torch::kFloat);
    at::Tensor input_cast = self.half_precision ? input.to(torch::kHalf) : input.to(torch::kFloat);
    at::Tensor weight_cast = self.half_precision ? weight.to(torch::kHalf) : weight.to(torch::kFloat);
    at::Tensor dx = at::empty_like(input_cast);
    at::Tensor contrib = at::empty_like(input_cast);

    [encoder setBuffer:(__bridge id<MTLBuffer>)dy_cast.storage().data_ptr().get() offset:dy_cast.storage_offset() atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input_cast.storage().data_ptr().get() offset:input_cast.storage_offset() atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)dx.storage().data_ptr().get() offset:dx.storage_offset() atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)contrib.storage().data_ptr().get() offset:contrib.storage_offset() atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight_cast.storage().data_ptr().get() offset:weight_cast.storage_offset() atIndex:4];

    uint num_rows = static_cast<uint>(input.size(0));
    uint u_dim = static_cast<uint>(self.dim);
    [encoder setBytes:&num_rows length:sizeof(uint) atIndex:5];
    [encoder setBytes:&u_dim length:sizeof(uint) atIndex:6];
    float eps_f = self.eps;
    [encoder setBytes:&eps_f length:sizeof(float) atIndex:7];

    NSUInteger max_threads = self.pipelineStateBackward.maxTotalThreadsPerThreadgroup;
    NSUInteger min_threads_per_row = 32;
    NSUInteger threads_per_row = std::max(min_threads_per_row, (((NSUInteger)self.dim + 31) / 32 * 32));
    if (threads_per_row > max_threads) threads_per_row = max_threads;
    NSUInteger max_rows_per_tg = max_threads / threads_per_row;
    NSUInteger rows_per_tg = std::min(max_rows_per_tg, (NSUInteger)num_rows);
    if (rows_per_tg == 0) {
        rows_per_tg = 1;
        threads_per_row = max_threads;
    }
    NSUInteger grid_y = (num_rows + rows_per_tg - 1) / rows_per_tg;

    uint u_threads_per_row = static_cast<uint>(threads_per_row);
    uint u_rows_per_tg = static_cast<uint>(rows_per_tg);
    [encoder setBytes:&u_threads_per_row length:sizeof(uint) atIndex:8];
    [encoder setBytes:&u_rows_per_tg length:sizeof(uint) atIndex:9];

    size_t type_size = 4;  // always float for shared mem
    NSUInteger num_sgs = (threads_per_row + 31) / 32;
    [encoder setThreadgroupMemoryLength:type_size * rows_per_tg * num_sgs atIndex:0];

    MTLSize threadgroup_size = MTLSizeMake(threads_per_row, rows_per_tg, 1);
    MTLSize grid_size = MTLSizeMake(1, grid_y, 1);

    [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];

    // Second pass: reduce contrib [num_rows, dim] across rows into dw [dim]
    at::Tensor dw = at::empty({self.dim}, torch::TensorOptions().device(input.device()).dtype(torch::kFloat));
    id<MTLComputeCommandEncoder> red = mpsStream->commandEncoder();
    [red setComputePipelineState:self.pipelineStateReduce];
    [red setBuffer:(__bridge id<MTLBuffer>)contrib.storage().data_ptr().get() offset:contrib.storage_offset() atIndex:0];
    [red setBuffer:(__bridge id<MTLBuffer>)dw.storage().data_ptr().get() offset:dw.storage_offset() atIndex:1];
    uint num_rows_red = static_cast<uint>(input.size(0));
    uint u_dim_red = static_cast<uint>(self.dim);
    [red setBytes:&num_rows_red length:sizeof(uint) atIndex:2];
    [red setBytes:&u_dim_red length:sizeof(uint) atIndex:3];
    NSUInteger tg = 256;
    if (tg > self.pipelineStateReduce.maxTotalThreadsPerThreadgroup) tg = self.pipelineStateReduce.maxTotalThreadsPerThreadgroup;
    NSUInteger gx = (self.dim + tg - 1) / tg;
    MTLSize tg_size = MTLSizeMake(tg, 1, 1);
    MTLSize gr_size = MTLSizeMake(gx * tg, 1, 1);
    [red dispatchThreads:gr_size threadsPerThreadgroup:tg_size];

    return {dx, dw};
}

@end


// C-API functions exposed to Python via PyTorch's C++ extension mechanism.
extern "C" {
void* rms_norm_new(int dim, double eps, bool half) {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Failed to get default Metal device.");
        }
        CustomRMSNormImpl* impl = [[CustomRMSNormImpl alloc] initWithDevice:device dim:dim eps:static_cast<float>(eps) half:half];
        return (__bridge_retained void*)impl;
    } catch (const std::exception& e) {
        std::cerr << "Error in rms_norm_new: " << e.what() << std::endl;
        return nullptr;
    }
}

void rms_norm_free(void* ptr) {
    if (ptr) {
        // Transfer ownership back to ARC and let it release the object.
        CustomRMSNormImpl* impl = (__bridge_transfer CustomRMSNormImpl*)ptr;
        impl = nil;
    }
}
}

at::Tensor rms_norm_forward(void* ptr, at::Tensor& input, at::Tensor& weight) {
    TORCH_CHECK(ptr, "CustomRMSNormImpl pointer is null.");
    CustomRMSNormImpl* impl = (__bridge CustomRMSNormImpl*)ptr;
    return [impl forward:input weight:weight];
}

std::pair<at::Tensor, at::Tensor> rms_norm_backward(void* ptr, at::Tensor& grad_output, at::Tensor& input, at::Tensor& weight) {
    TORCH_CHECK(ptr, "CustomRMSNormImpl pointer is null.");
    CustomRMSNormImpl* impl = (__bridge CustomRMSNormImpl*)ptr;
    return [impl backward:grad_output input:input weight:weight];
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_new", [](int dim, double eps, bool half_precision) {
        return rms_norm_new(dim, eps, half_precision);
    });
    m.def("rms_norm_free", &rms_norm_free);
    m.def("rms_norm_forward", [](void* ptr, at::Tensor input, at::Tensor weight) {
        return rms_norm_forward(ptr, input, weight);
    });
    m.def("rms_norm_backward", [](void* ptr, at::Tensor grad_output, at::Tensor input, at::Tensor weight) -> std::pair<at::Tensor, at::Tensor> {
        return rms_norm_backward(ptr, grad_output, input, weight);
    });
}