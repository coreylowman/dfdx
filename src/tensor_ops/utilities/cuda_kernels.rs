use crate::{
    shapes::{Dtype, Shape},
    tensor::{cuda::Cuda, Tensor},
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
    unique_id::unique_id,
};
use cudarc::driver::{AsKernelParam, CudaSlice, LaunchAsync, LaunchConfig};
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

macro_rules! cuda_unary {
    ($Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel<$TypeName> for $Op {
            const PTX_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
}

pub(crate) use cuda_unary;

impl<E: Dtype, K: UnaryOpCudaKernel<E> + AsKernelParam> UnaryKernel<K, E> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: K,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        if !self.dev.has_func(K::MODULE_NAME, K::FWD_FN_NAME) {
            self.dev
                .load_ptx(K::PTX_SRC.into(), K::MODULE_NAME, &K::ALL_FN_NAMES)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.dev.alloc_async::<E>(numel) }?;

        let fwd_fn = self.dev.get_func(K::MODULE_NAME, K::FWD_FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (op, numel, inp.data.as_ref(), &mut storage);
        unsafe { fwd_fn.launch_async(cfg, params) }?;

        Ok(Tensor {
            id: unique_id(),
            data: Arc::new(storage),
            shape: inp.shape,
            strides: inp.strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let bwd_fn = self.dev.get_func(K::MODULE_NAME, K::BWD_FN_NAME).unwrap();
        let numel = inp.data.len();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (op, numel, inp.data.as_ref(), grad_inp, grad_out);
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

macro_rules! cuda_binary {
    ($Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel<$TypeName> for $Op {
            const PTX_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
}

pub(crate) use cuda_binary;

impl<E: Dtype, K: BinaryOpCudaKernel<E> + AsKernelParam> BinaryKernel<K, E> for Cuda {
    fn forward<S: Shape>(
        &self,
        op: K,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
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
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(Tensor {
            id: unique_id(),
            data: Arc::new(storage),
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        lhs: &Tensor<S, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<S, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let bwd_fn = self.dev.get_func(K::MODULE_NAME, K::BWD_FN_NAME).unwrap();
        let numel = lhs.shape.num_elements();

        let dims: CudaSlice<usize> = self.dev.take_async(lhs.shape.concrete().into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.take_async(lhs.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.take_async(rhs.strides.into())?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            op,
            numel,             // const size_t numel,
            S::NUM_DIMS,       // const size_t num_dims,
            &dims,             // const size_t *dims,
            lhs.data.as_ref(), // const float *lhs,
            grad_lhs,          // float *grad_lhs,
            &lhs_strides,      // const size_t *lhs_strides,
            rhs.data.as_ref(), // const float *rhs,
            grad_rhs,          // float *grad_rhs,
            &rhs_strides,      // const size_t *rhs_strides,
            grad_out,          // const float *grad_out,
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
