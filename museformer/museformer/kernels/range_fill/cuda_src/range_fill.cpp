// #pragma once
#include <torch/extension.h>


// CUDA函数声明
int cudaForwardLauncher(
    const at::Tensor&,
    const at::Tensor&,
    const long,
    at::Tensor&
);


// C++函数包装
int cuda_forward(const at::Tensor& ranges,
                 const at::Tensor& values,
                 const long num_chunks,
                 at::Tensor& output)  {
    at::DeviceGuard guard(ranges.device());
    cudaForwardLauncher(ranges, values, num_chunks, output);
    return 0;
}

// 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cuda_forward", &cuda_forward, "");
}