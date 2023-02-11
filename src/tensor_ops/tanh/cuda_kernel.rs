use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::TanhKernelOp {}

const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/tanh.ptx"));

impl UnaryOpCudaKernel<f32> for super::TanhKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "tanh_f32";
    const FWD_FN_NAME: &'static str = "tanh_forward_f32";
    const BWD_FN_NAME: &'static str = "tanh_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::TanhKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "tanh_f64";
    const FWD_FN_NAME: &'static str = "tanh_forward_f64";
    const BWD_FN_NAME: &'static str = "tanh_backward_f64";
}
