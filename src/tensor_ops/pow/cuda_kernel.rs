use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::device::AsKernelParam for super::PowKernelOp<f32> {}

impl UnaryOpCudaKernel for super::PowKernelOp<f32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/powf.ptx"));
    const MODULE_NAME: &'static str = "powf";
    const FWD_FN_NAME: &'static str = "powf_forward";
    const BWD_FN_NAME: &'static str = "powf_backward";
}

unsafe impl cudarc::device::AsKernelParam for super::PowKernelOp<i32> {}

impl UnaryOpCudaKernel for super::PowKernelOp<i32> {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/powi.ptx"));
    const MODULE_NAME: &'static str = "powi";
    const FWD_FN_NAME: &'static str = "powi_forward";
    const BWD_FN_NAME: &'static str = "powi_backward";
}
