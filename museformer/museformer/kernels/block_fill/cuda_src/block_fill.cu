//#include <cstdio>
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
    const scalar_t * block_ranges,
    const bool * head_masks,
    const long input_size,
    const long num_lines, const long max_query, const long max_key, const long num_heads,
    const long seq_len_2,
    const long line_stride, const long query_stride, const long key_stride, const long seq_len_square,
    const bool fill_value,
    bool * out) {

    // block_ranges: (num_lines, 4)
    // head_masks: (num_lines, num_heads)
    // input_size == num_lines * max_query * max_key * num_heads
    // line_stride == max_query * max_key * num_heads
    // query_stride == max_key * num_heads
    // key_stride == num_heads
    // out: (num_heads, seq_len_1, seq_len_2)

    const long index = long(blockIdx.x) * long(blockDim.x) + long(threadIdx.x);

    if (index >= input_size) return;

    const long line_index = index / line_stride;
    assert (line_index < num_lines);
    const long query_index = (index % line_stride) / query_stride;
    assert (query_index < max_query);
    const long key_index = (index % query_stride) / key_stride;
    assert (key_index < max_key);
    const long head_index = (index % key_stride);

    const long line_start = line_index * 4;

    const long query_begin = block_ranges[line_start];
    const long query_end = block_ranges[line_start + 1];
    const long query_index_end = query_index + query_begin;
    if (!(query_index_end < query_end)) return;

    const long key_begin = block_ranges[line_start + 2];
    const long key_end = block_ranges[line_start + 3];
    const long key_index_end = key_index + key_begin;
    if (!(key_index_end < key_end)) return;

    if (head_masks[line_index * num_heads + head_index])
        out[head_index * seq_len_square + query_index_end * seq_len_2 + key_index_end] = fill_value;
}


int cudaForwardLauncher(
    const at::Tensor& block_ranges,
    const at::Tensor& head_masks,
    const long num_lines,
    const long max_query,
    const long max_key,
    const long num_heads,
    const long seq_len_1,
    const long seq_len_2,
    const bool fill_value,
    at::Tensor& out
) {
    const long input_size = num_lines * max_query * max_key * num_heads;
    assert (input_size <= THREADS_PER_BLOCK * MAX_GRID_NUM);

    const long key_stride = num_heads;
    const long query_stride = max_key * key_stride;
    const long line_stride = max_query * query_stride;
    const long seq_len_square = seq_len_1 * seq_len_2;

    AT_DISPATCH_INTEGRAL_TYPES(
        block_ranges.type(), "cudaForward",
        ([&] {
            const scalar_t *block_ranges_ = block_ranges.data_ptr<scalar_t>();
            const bool *head_masks_ = head_masks.data_ptr<bool>();
            bool *out_ = out.data_ptr<bool>();

            cudaForward<<<GET_BLOCKS(input_size), THREADS_PER_BLOCK>>>(
                block_ranges_, head_masks_, input_size,
                num_lines, max_query, max_key, num_heads, seq_len_2,
                line_stride, query_stride, key_stride, seq_len_square,
                fill_value,
                out_
            );
          }
        )
    );

    THCudaCheck(cudaGetLastError());

    return 0;
}
