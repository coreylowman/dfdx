use crate::{
    shapes::{Dtype, Shape},
    tensor::{launch_cfg, unique_id, Cuda, Tensor},
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use cudarc::driver::{CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync};
use std::{sync::Arc, vec::Vec};

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

impl<E: Dtype, K: UnaryOpCudaKernel<E> + DeviceRepr> UnaryKernel<K, E> for Cuda {
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
        let mut storage = unsafe { self.dev.alloc::<E>(numel) }?;

        let fwd_fn = self.dev.get_func(K::MODULE_NAME, K::FWD_FN_NAME).unwrap();
        let cfg = launch_cfg(numel as u32);
        let params = (op, numel, inp.data.as_ref(), &mut storage);
        unsafe { fwd_fn.launch(cfg, params) }?;

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
        let cfg = launch_cfg(numel as u32);
        let params = (op, numel, inp.data.as_ref(), grad_inp, grad_out);
        unsafe { bwd_fn.launch(cfg, params) }?;
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
    const BWD_LHS_FN_NAME: &'static str;

    /// Name of function in the .cu file
    const BWD_RHS_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 3] = [
        Self::FWD_FN_NAME,
        Self::BWD_LHS_FN_NAME,
        Self::BWD_RHS_FN_NAME,
    ];
}

macro_rules! cuda_binary {
    ($Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel<$TypeName> for $Op {
            const PTX_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
}

/// Similar to [permute_for_reductions], but defines the summed axes as the broadcasted axes in
/// arg2_strides, and orders non-summed axes so that the output of chunk_sum can be read correctly
/// using arg2_strides.
fn permute_for_binary_backward<I>(
    out_dims: I,
    out_strides: I,
    arg1_strides: I,
    arg2_strides: I,
) -> ((Vec<usize>, Vec<usize>), Vec<usize>)
where
    I: IntoIterator<Item = usize>,
{
    let mut tmp: Vec<_> = out_dims
        .into_iter()
        .zip(out_strides.into_iter())
        .zip(arg1_strides.into_iter())
        .zip(arg2_strides.into_iter())
        .map(|(x, out_stride)| {
            let ord = if out_stride == 0 {
                (true, -(x.0 .1 as isize))
            } else {
                (false, -(out_stride as isize))
            };

            (ord, x)
        })
        .collect();

    tmp.sort_unstable_by_key(|(ord, _)| *ord);

    tmp.into_iter().map(|(_ord, x)| x).unzip()
}

pub(crate) use cuda_binary;

impl<E: Dtype, K: BinaryOpCudaKernel<E> + DeviceRepr + Clone> BinaryKernel<K, E> for Cuda {
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

        let fwd_fn = self.dev.get_func(K::MODULE_NAME, K::FWD_FN_NAME).unwrap();
        let cfg = launch_cfg(numel as u32);

        let mut info: Vec<usize> = Vec::with_capacity(3 * S::NUM_DIMS);
        info.extend(shape.concrete());
        info.extend(lhs.strides);
        info.extend(rhs.strides);

        let mut storage = unsafe { self.dev.alloc::<E>(numel) }?;
        let info: CudaSlice<usize> = self.dev.htod_copy(info)?;
        let params = (
            op,
            numel,             // const size_t numel,
            S::NUM_DIMS,       // const size_t num_dims,
            &info,             // const size_t *dims,
            lhs.data.as_ref(), // const float *lhs,
            rhs.data.as_ref(), // const float *rhs,
            &mut storage,      // float *out,
        );
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(Tensor {
            id: unique_id(),
            data: Arc::new(storage),
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }

    // NOTE: if it becomes possible for grad_out to be broadcasted, (i.e. if #366 is resolved), we
    // need to pass an elems_per_thread argument to the backward cuda kernels, as we do in sum_to.
    fn backward<S: Shape>(
        &self,
        op: K,
        lhs: &Tensor<S, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<S, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let bwd_lhs_fn = self
            .dev
            .get_func(K::MODULE_NAME, K::BWD_LHS_FN_NAME)
            .unwrap();

        let bwd_rhs_fn = self
            .dev
            .get_func(K::MODULE_NAME, K::BWD_RHS_FN_NAME)
            .unwrap();

        let numel = lhs.shape.num_elements();
        let cfg = launch_cfg(numel as u32);

        let ((out_dims1, out_strides1), rhs_strides1) = permute_for_binary_backward(
            lhs.shape.concrete(),
            lhs.shape.strides(),
            rhs.strides,
            lhs.strides,
        );

        let ((out_dims2, out_strides2), lhs_strides2) = permute_for_binary_backward(
            lhs.shape.concrete(),
            lhs.shape.strides(),
            lhs.strides,
            rhs.strides,
        );

        let mut info: Vec<usize> = Vec::with_capacity(6 * S::NUM_DIMS);
        info.extend(out_dims1);
        info.extend(out_strides1);
        info.extend(rhs_strides1);
        info.extend(out_dims2);
        info.extend(out_strides2);
        info.extend(lhs_strides2);

        let info = self.dev.htod_copy(info)?;

        self.par_stream.wait_for_default()?;

        let params_lhs = (
            op.clone(),  // const OP_STRUCT op,
            numel,       // const size_t numel,
            S::NUM_DIMS, // const size_t num_dims,
            &info,
            lhs.data.as_ref(),      // const TYPENAME *lhs,
            grad_lhs,               // TYPENAME *grad_lhs,
            numel / lhs.data.len(), // const size_t chunk_len,
            rhs.data.as_ref(),      // const TYPENAME *rhs,
            grad_out,               // const TYPENAME *grad_out
        );

        unsafe { bwd_lhs_fn.launch_on_stream(&self.par_stream, cfg, params_lhs) }?;

        let params_rhs = (
            op,          // const OP_STRUCT op,
            numel,       // const size_t numel,
            S::NUM_DIMS, // const size_t num_dims,
            &info,
            lhs.data.as_ref(),      // const TYPENAME *lhs,
            rhs.data.as_ref(),      // const TYPENAME *rhs,
            grad_rhs,               // TYPENAME *grad_rhs,
            numel / rhs.data.len(), // const size_t chunk_len,
            grad_out,               // const TYPENAME *grad_out
        );

        unsafe { bwd_rhs_fn.launch(cfg, params_rhs) }?;

        self.dev.wait_for(&self.par_stream)?;

        Ok(())
    }
}
