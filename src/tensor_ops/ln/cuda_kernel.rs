use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::LnKernelOp {}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/ln.ptx"));

impl UnaryOpCudaKernel<f32> for super::LnKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "ln_f32";
    const FWD_FN_NAME: &'static str = "ln_forward_f32";
    const BWD_FN_NAME: &'static str = "ln_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::LnKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "ln_f64";
    const FWD_FN_NAME: &'static str = "ln_forward_f64";
    const BWD_FN_NAME: &'static str = "ln_backward_f64";
}
