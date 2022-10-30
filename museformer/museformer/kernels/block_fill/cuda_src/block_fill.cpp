// #pragma once
#include <torch/extension.h>


// CUDA函数声明
int cudaForwardLauncher(
    const at::Tensor&, const at::Tensor&,
    const long, const long, const long, const long, const long, const long, const bool, at::Tensor&
);


// C++函数包装
int cuda_forward(const at::Tensor& block_ranges,
                 const at::Tensor& head_masks,
                 const long num_lines,
                 const long max_query,
                 const long max_key,
                 const long num_heads,
                 const long seq_len_1,
                 const long seq_len_2,
                 const bool fill_value,
                 at::Tensor& out)  {
    at::DeviceGuard guard(block_ranges.device());
    cudaForwardLauncher(block_ranges, head_masks, num_lines, max_query, max_key, num_heads,
    seq_len_1, seq_len_2, fill_value, out);
    return 0;
}

// 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cuda_forward", &cuda_forward, "cuda_forward");
}