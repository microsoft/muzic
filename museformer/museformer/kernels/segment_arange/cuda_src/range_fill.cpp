// #pragma once
#include <torch/extension.h>


// CUDA函数声明
int cudaForwardLauncher(
    const at::Tensor&,
    const long,
    const long,
    at::Tensor&
);


// C++函数包装
int cuda_forward(const at::Tensor& ranges,
                 const long start,
                 const long num_chunks,
                 at::Tensor& output)  {
    at::DeviceGuard guard(ranges.device());
    cudaForwardLauncher(ranges, start, num_chunks, output);
    return 0;
}

// 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cuda_forward", &cuda_forward, "");
}