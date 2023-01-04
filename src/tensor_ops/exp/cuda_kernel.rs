use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::device::IntoKernelParam for super::ExpKernelOp {
    #[inline(always)]
    fn into_kernel_param(self) -> *mut std::ffi::c_void {
        (&self) as *const Self as *mut std::ffi::c_void
    }
}

impl UnaryOpCudaKernel for super::ExpKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/exp.ptx"));
    const MODULE_NAME: &'static str = "exp";
    const FWD_FN_NAME: &'static str = "exp_forward";
    const BWD_FN_NAME: &'static str = "exp_backward";
}
