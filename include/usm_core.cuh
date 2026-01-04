#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

namespace usm {

/**
 * USM: Universal Streaming Mechanism
 * ----------------------------------
 * A paradigm shift from "Thread Ownership" to "Holistic Traversal".
 * Instead of assigning threads to specific streams, USM threads traverse 
 * the entire data field in a grid-stride pattern.
 */

template <typename Navigator, typename Operator>
__global__ void __launch_bounds__(256) usm_kernel(int total_elements, Navigator nav, Operator op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop
    for (; idx < total_elements; idx += stride) {
        auto context = nav.get_context(idx);
        op.apply(idx, context);
    }
}

template <typename Navigator, typename Operator>
float launch(int total_elements, Navigator nav, Operator op, cudaStream_t stream = 0) {
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 2048) grid_size = 2048; // Saturation heuristic

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    usm_kernel<<<grid_size, block_size, 0, stream>>>(total_elements, nav, op);
    cudaEventRecord(stop, stream);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1.0f;

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    return milliseconds;
}
} // namespace usm