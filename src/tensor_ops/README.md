# dfdx::tensor_ops

This module is organized as follows:
- Each individual op has its own module
    - If it requires its own kernels, its in a directory (e.g. abs/)
    - If it's made up of other tensor ops, its in a file (e.g. softmax.rs)
- Ops with their own kernels contain the following files:
    - `mod.rs` - contains Tensor method/function, docstrings, and tests
    - `cpu_kernel.rs` contains the implementation for the CPU
    - `cuda_kernel.rs` contains the code for invoking a cuda kernel
    - `<tensor_op>.cu` contains the actual cuda kernel logic
