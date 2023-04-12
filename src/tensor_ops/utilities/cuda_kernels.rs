use crate::{
    shapes::{Dtype, Shape},
    tensor::*,
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync};
use std::{borrow::Cow, sync::Arc, vec::Vec};

pub trait UnaryOpCudaKernel<E> {
    const DF_USES_FX: bool;
    const HAS_CONST_DF: bool;

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
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = false;
            const PTX_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (df(f(x)) $Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = true;
            const HAS_CONST_DF: bool = false;
            const PTX_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = true;
            const PTX_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
}

pub(crate) use cuda_unary;

impl<E: Dtype, K: UnaryOpCudaKernel<E> + DeviceRepr> UnaryKernel<K, E> for Cuda {
    const BACKWARD_WITHOUT_INP: bool = K::DF_USES_FX;
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: K,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        if !self.dev.has_func(K::MODULE_NAME, K::FWD_FN_NAME) {
            self.dev
                .load_ptx(K::PTX_SRC.into(), K::MODULE_NAME, &K::ALL_FN_NAMES)?;
        }

        let fwd_fn = self.dev.get_func(K::MODULE_NAME, K::FWD_FN_NAME).unwrap();

        match inp {
            Cow::Borrowed(inp) => {
                let numel = inp.data.len();
                let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;

                let cfg = launch_cfg::<128>(numel as u32);
                let params = (op, numel, inp.data.as_ref(), &mut storage);
                unsafe { fwd_fn.launch(cfg, params) }?;
                Ok(self.build_tensor(inp.shape, inp.strides, storage))
            }
            Cow::Owned(mut inp) => {
                inp.id = unique_id();
                let numel = inp.data.len();
                let cfg = launch_cfg::<128>(numel as u32);
                let params = (op, numel, 0u64, Arc::make_mut(&mut inp.data));
                unsafe { fwd_fn.launch(cfg, params) }?;
                Ok(inp)
            }
        }
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let bwd_fn = self.dev.get_func(K::MODULE_NAME, K::BWD_FN_NAME).unwrap();
        match (inp.data(), out.data()) {
            (None, None) => {
                let cfg = launch_cfg::<128>(inp.len() as u32);
                let params = (op, inp.len(), 0u64, grad_inp, 0u64, grad_out);
                unsafe { bwd_fn.launch(cfg, params) }?;
            }
            (None, Some(out_buf)) => {
                let cfg = launch_cfg::<128>(inp.len() as u32);
                let params = (op, inp.len(), 0u64, grad_inp, out_buf, grad_out);
                unsafe { bwd_fn.launch(cfg, params) }?;
            }
            (Some(inp_buf), None) => {
                let numel = inp.len();
                let cfg = launch_cfg::<128>(numel as u32);
                let params = (op, numel, inp_buf, grad_inp, 0u64, grad_out);
                unsafe { bwd_fn.launch(cfg, params) }?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

pub trait BinaryOpCudaKernel<E> {
    const HAS_CONST_DF: bool;

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
            const HAS_CONST_DF: bool = false;
            const PTX_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::cuda_kernels::BinaryOpCudaKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = true;
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
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: K,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        if !self.dev.has_func(K::MODULE_NAME, K::FWD_FN_NAME) {
            self.dev
                .load_ptx(K::PTX_SRC.into(), K::MODULE_NAME, &K::ALL_FN_NAMES)?;
        }
        let fwd_fn = self.dev.get_func(K::MODULE_NAME, K::FWD_FN_NAME).unwrap();

        let shape = match &lhs {
            Cow::Borrowed(lhs) => lhs.shape,
            Cow::Owned(lhs) => lhs.shape,
        };
        let strides = shape.strides();
        let numel = shape.num_elements();
        let cfg = launch_cfg::<128>(numel as u32);

        let lhs_strides = match &lhs {
            Cow::Borrowed(lhs) => lhs.strides,
            Cow::Owned(lhs) => lhs.strides,
        };
        let rhs_strides = match &rhs {
            Cow::Borrowed(rhs) => rhs.strides,
            Cow::Owned(rhs) => rhs.strides,
        };

        let mut info: Vec<usize> = Vec::with_capacity(3 * S::NUM_DIMS);
        info.extend(shape.concrete());
        info.extend(lhs_strides);
        info.extend(rhs_strides);
        let info = self.dev.htod_copy(info)?;

        match (lhs, rhs) {
            (Cow::Borrowed(lhs), Cow::Borrowed(rhs)) => {
                let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;
                let params = (
                    op,
                    numel,             // const size_t numel,
                    S::NUM_DIMS,       // const size_t num_dims,
                    &info,             // const size_t *info,
                    lhs.data.as_ref(), // const float *lhs,
                    rhs.data.as_ref(), // const float *rhs,
                    &mut storage,      // float *out,
                );
                unsafe { fwd_fn.launch(cfg, params) }?;
                Ok(self.build_tensor(shape, strides, storage))
            }
            (Cow::Owned(mut lhs), Cow::Owned(mut rhs)) => {
                let lhs_valid = lhs.strides == lhs.shape.strides();
                let rhs_valid = rhs.strides == rhs.shape.strides();
                if lhs_valid || rhs_valid {
                    let lhs_count = std::sync::Arc::strong_count(&lhs.data);
                    let rhs_count = std::sync::Arc::strong_count(&rhs.data);
                    if rhs_valid && (rhs_count == 1 || !lhs_valid || lhs_count != 1) {
                        rhs.id = unique_id();
                        let params = (
                            op,
                            numel,
                            S::NUM_DIMS,
                            &info,
                            lhs.data.as_ref(),
                            0u64,
                            Arc::make_mut(&mut rhs.data),
                        );
                        unsafe { fwd_fn.launch(cfg, params) }?;
                        Ok(rhs)
                    } else {
                        lhs.id = unique_id();
                        let params = (
                            op,
                            numel,                        // const size_t numel,
                            S::NUM_DIMS,                  // const size_t num_dims,
                            &info,                        // const size_t *info,
                            0u64,                         // const float *lhs,
                            rhs.data.as_ref(),            // const float *rhs,
                            Arc::make_mut(&mut lhs.data), // float *out,
                        );
                        unsafe { fwd_fn.launch(cfg, params) }?;
                        Ok(lhs)
                    }
                } else {
                    let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;
                    let params = (
                        op,
                        numel,             // const size_t numel,
                        S::NUM_DIMS,       // const size_t num_dims,
                        &info,             // const size_t *info,
                        lhs.data.as_ref(), // const float *lhs,
                        rhs.data.as_ref(), // const float *rhs,
                        &mut storage,      // float *out,
                    );
                    unsafe { fwd_fn.launch(cfg, params) }?;
                    Ok(self.build_tensor(shape, strides, storage))
                }
            }
            _ => unreachable!(),
        }
    }

    // NOTE: if it becomes possible for grad_out to be broadcasted, (i.e. if #366 is resolved), we
    // need to pass an elems_per_thread argument to the backward cuda kernels, as we do in sum_to.
    fn backward<S: Shape>(
        &self,
        op: K,
        lhs: &impl Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &impl Tensorlike<S, E, Self>,
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

        let shape = lhs.shape();
        let (lhs_strides, lhs_len) = (lhs.strides(), lhs.len());
        let (rhs_strides, rhs_len) = (rhs.strides(), rhs.len());

        let numel = shape.num_elements();
        let cfg = launch_cfg::<128>(numel as u32);

        let ((out_dims1, out_strides1), rhs_strides1) = permute_for_binary_backward(
            shape.concrete(),
            shape.strides(),
            rhs_strides,
            lhs_strides,
        );

        let ((out_dims2, out_strides2), lhs_strides2) = permute_for_binary_backward(
            shape.concrete(),
            shape.strides(),
            lhs_strides,
            rhs_strides,
        );

        let mut info: Vec<usize> = Vec::with_capacity(6 * S::NUM_DIMS);
        info.extend(out_dims1);
        info.extend(out_strides1);
        info.extend(rhs_strides1);
        info.extend(out_dims2);
        info.extend(out_strides2);
        info.extend(lhs_strides2);
        let info = self.dev.htod_copy(info)?;

        match (lhs.data(), rhs.data()) {
            (Some(lhs_buf), Some(rhs_buf)) => {
                let params_lhs = (
                    op.clone(),      // const OP_STRUCT op,
                    numel,           // const size_t numel,
                    S::NUM_DIMS,     // const size_t num_dims,
                    &info,           // const size_t *info,
                    lhs_buf,         // const TYPENAME *lhs,
                    grad_lhs,        // TYPENAME *grad_lhs,
                    numel / lhs_len, // const size_t chunk_len,
                    rhs_buf,         // const TYPENAME *rhs,
                    grad_out,        // const TYPENAME *grad_out
                );
                let params_rhs = (
                    op,              // const OP_STRUCT op,
                    numel,           // const size_t numel,
                    S::NUM_DIMS,     // const size_t num_dims,
                    &info,           // const size_T * info,
                    lhs_buf,         // const TYPENAME *lhs,
                    rhs_buf,         // const TYPENAME *rhs,
                    grad_rhs,        // TYPENAME *grad_rhs,
                    numel / rhs_len, // const size_t chunk_len,
                    grad_out,        // const TYPENAME *grad_out
                );

                self.par_stream.wait_for_default()?;
                unsafe { bwd_lhs_fn.launch_on_stream(&self.par_stream, cfg, params_lhs) }?;
                unsafe { bwd_rhs_fn.launch(cfg, params_rhs) }?;
                self.dev.wait_for(&self.par_stream)?;
            }
            (None, None) => {
                let params_lhs = (
                    op.clone(),      // const OP_STRUCT op,
                    numel,           // const size_t numel,
                    S::NUM_DIMS,     // const size_t num_dims,
                    &info,           // const size_t *info,
                    0u64,            // const TYPENAME *lhs,
                    grad_lhs,        // TYPENAME *grad_lhs,
                    numel / lhs_len, // const size_t chunk_len,
                    0u64,            // const TYPENAME *rhs,
                    grad_out,        // const TYPENAME *grad_out
                );
                let params_rhs = (
                    op,              // const OP_STRUCT op,
                    numel,           // const size_t numel,
                    S::NUM_DIMS,     // const size_t num_dims,
                    &info,           // const size_T * info,
                    0u64,            // const TYPENAME *lhs,
                    0u64,            // const TYPENAME *rhs,
                    grad_rhs,        // TYPENAME *grad_rhs,
                    numel / rhs_len, // const size_t chunk_len,
                    grad_out,        // const TYPENAME *grad_out
                );

                self.par_stream.wait_for_default()?;
                unsafe { bwd_lhs_fn.launch_on_stream(&self.par_stream, cfg, params_lhs) }?;
                unsafe { bwd_rhs_fn.launch(cfg, params_rhs) }?;
                self.dev.wait_for(&self.par_stream)?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}
