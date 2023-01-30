use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::CosKernelOp {}

impl UnaryOpCudaKernel<f32> for super::CosKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/cos.ptx"));
    const MODULE_NAME: &'static str = "cos";
    const FWD_FN_NAME: &'static str = "cos_forward_f32";
    const BWD_FN_NAME: &'static str = "cos_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::CosKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/cos.ptx"));
    const MODULE_NAME: &'static str = "cos";
    const FWD_FN_NAME: &'static str = "cos_forward_f64";
    const BWD_FN_NAME: &'static str = "cos_backward_f64";
}
