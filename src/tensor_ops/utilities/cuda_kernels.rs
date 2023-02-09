use crate::{
    shapes::{Dtype, Shape},
    tensor::cuda::{Cuda, CudaArray},
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use cudarc::driver::{AsKernelParam, CudaSlice, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use std::sync::Arc;

pub trait UnaryOpCudaKernel<E> {
    /// Compiled by build.rs
    const PTX_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .cu file
    const FWD_FN_NAME: &'static str;

    /// Name of function in the .cu file
    const BWD_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 2] = [Self::FWD_FN_NAME, Self::BWD_FN_NAME];
}

impl<E: Dtype, K: UnaryOpCudaKernel<E> + AsKernelParam> UnaryKernel<K, E>
    for Cuda
{
    fn forward<S: Shape>(
        &self,
        op: K,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        if !self.dev.has_func(K::MODULE_NAME, K::FWD_FN_NAME) {
            self.dev
                .load_ptx(K::PTX_SRC.into(), K::MODULE_NAME, &K::ALL_FN_NAMES)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.dev.alloc_async::<E>(numel) }?;

        let fwd_fn = self.dev.get_func(K::MODULE_NAME, K::FWD_FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            op,
            numel,             // const size_t numel,
            inp.data.as_ref(), // const float *inp,
            &mut storage,      // float *out
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;

        Ok(CudaArray {
            data: Arc::new(storage),
            shape: inp.shape,
            strides: inp.strides,
        })
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        inp: &Self::Storage<S, E>,
        grad_inp: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        let bwd_fn = self.dev.get_func(K::MODULE_NAME, K::BWD_FN_NAME).unwrap();
        let numel = inp.data.len();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            op,
            numel,                             // const size_t numel,
            inp.data.as_ref(),                 // const float *inp,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            grad_out.data.as_ref(),            // const float *grad_out
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}

pub trait BinaryOpCudaKernel<E> {
    /// Compiled by build.rs
    const PTX_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .cu file
    const FWD_FN_NAME: &'static str;

    /// Name of function in the .cu file
    const BWD_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 2] = [Self::FWD_FN_NAME, Self::BWD_FN_NAME];
}

impl<E: Dtype, K: BinaryOpCudaKernel<E> + AsKernelParam> BinaryKernel<K, E>
    for Cuda
{
    fn forward<S: Shape>(
        &self,
        op: K,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        if !self.dev.has_func(K::MODULE_NAME, K::FWD_FN_NAME) {
            self.dev
                .load_ptx(K::PTX_SRC.into(), K::MODULE_NAME, &K::ALL_FN_NAMES)?;
        }

        let shape = lhs.shape;
        let strides = lhs.shape.strides();
        let numel = shape.num_elements();

        let mut storage = unsafe { self.dev.alloc_async::<E>(numel) }?;

        let dims: CudaSlice<usize> = self.dev.take_async(shape.concrete().into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.take_async(lhs.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.take_async(rhs.strides.into())?;
        let out_strides: CudaSlice<usize> = self.dev.take_async(strides.into())?;

        let fwd_fn = self.dev.get_func(K::MODULE_NAME, K::FWD_FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            op,
            numel,             // const size_t numel,
            S::NUM_DIMS,       // const size_t num_dims,
            &dims,             // const size_t *dims,
            lhs.data.as_ref(), // const float *lhs,
            &lhs_strides,      // const size_t *lhs_strides,
            rhs.data.as_ref(), // const float *rhs,
            &rhs_strides,      // const size_t *rhs_strides,
            &mut storage,      // float *out,
            &out_strides,      // const size_t *out_strides
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides,
        })
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        lhs: &Self::Storage<S, E>,
        grad_lhs: &mut Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
        grad_rhs: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        let bwd_fn = self.dev.get_func(K::MODULE_NAME, K::BWD_FN_NAME).unwrap();
        let numel = lhs.shape.num_elements();

        let dims: CudaSlice<usize> = self.dev.take_async(lhs.shape.concrete().into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.take_async(lhs.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.take_async(rhs.strides.into())?;
        let out_strides: CudaSlice<usize> = self.dev.take_async(grad_out.strides.into())?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            op,
            numel,                             // const size_t numel,
            S::NUM_DIMS,                       // const size_t num_dims,
            &dims,                             // const size_t *dims,
            lhs.data.as_ref(),                 // const float *lhs,
            Arc::make_mut(&mut grad_lhs.data), // float *grad_lhs,
            &lhs_strides,                      // const size_t *lhs_strides,
            rhs.data.as_ref(),                 // const float *rhs,
            Arc::make_mut(&mut grad_rhs.data), // float *grad_rhs,
            &rhs_strides,                      // const size_t *rhs_strides,
            grad_out.data.as_ref(),            // const float *grad_out,
            &out_strides,                      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
