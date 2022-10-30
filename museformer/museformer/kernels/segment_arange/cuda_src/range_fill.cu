#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>


const long THREADS_PER_BLOCK = 1024;
const long MAX_GRID_NUM = 2147483647;


inline long GET_BLOCKS(const long N) {
  long optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  return long(min(optimal_block_num, MAX_GRID_NUM));
}


template <typename scalar_t>
__global__ void cudaForward(
    const long * ranges,
    const long start,
    const long input_size,
    scalar_t * output) {

    const long index = long(blockIdx.x) * long(blockDim.x) + long(threadIdx.x);

    if (index >= input_size) return;

    const long line_start = index * 2;
    const long begin_idx = ranges[line_start];
    const long end_idx = ranges[line_start + 1];

    long cur = 0;
    for (long idx = begin_idx; idx < end_idx; idx++) {
        output[idx] = cur + start;
        cur++;
    }
}


int cudaForwardLauncher(
    const at::Tensor& ranges,
    const long start,
    const long num_chunks,
    at::Tensor& output
) {
    const long input_size = num_chunks;
    assert (input_size <= THREADS_PER_BLOCK * MAX_GRID_NUM);

    AT_DISPATCH_INTEGRAL_TYPES(
        ranges.type(), "cudaForward",
        ([&] {
            const long *ranges_ = ranges.data_ptr<long>();
            scalar_t *output_ = output.data_ptr<scalar_t>();

            cudaForward<<<GET_BLOCKS(input_size), THREADS_PER_BLOCK>>>(
                ranges_, start, input_size, output_
            );
          }
        )
    );

    THCudaCheck(cudaGetLastError());

    return 0;
}
