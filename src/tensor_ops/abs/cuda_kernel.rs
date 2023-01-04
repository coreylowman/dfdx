use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::device::IntoKernelParam for super::AbsKernelOp {
    #[inline(always)]
    fn into_kernel_param(self) -> *mut std::ffi::c_void {
        (&self) as *const Self as *mut std::ffi::c_void
    }
}

impl UnaryOpCudaKernel for super::AbsKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/abs.ptx"));
    const MODULE_NAME: &'static str = "abs";
    const FWD_FN_NAME: &'static str = "abs_forward";
    const BWD_FN_NAME: &'static str = "abs_backward";
}
