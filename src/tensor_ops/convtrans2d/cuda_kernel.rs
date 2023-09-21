use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::driver::{DeviceRepr, LaunchAsync, ValidAsZeroBits};

use crate::{
    dtypes::*,
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor, Tensorlike},
};

use std::sync::Arc;

unsafe impl DeviceRepr for super::ConvTrans2DOp {}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/convtrans2d.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

#[cfg(feature = "f16")]
impl HasCudaKernel<AMP<f16>> for Cuda {
    const MOD: &'static str = "convtrans2d_f16";
    const FNS: &'static [&'static str] = &[
        "unfold_input_into_patches_f16",
        "unfold_output_into_patches_f16",
        "transpose_filters_f16",
    ];
}

#[cfg(feature = "f16")]
impl HasCudaKernel<f16> for Cuda {
    const MOD: &'static str = "convtrans2d_f16";
    const FNS: &'static [&'static str] = &[
        "unfold_input_into_patches_f16",
        "unfold_output_into_patches_f16",
        "transpose_filters_f16",
    ];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "convtrans2d_f32";
    const FNS: &'static [&'static str] = &[
        "unfold_input_into_patches_f32",
        "unfold_output_into_patches_f32",
        "transpose_filters_f32",
    ];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "convtrans2d_f64";
    const FNS: &'static [&'static str] = &[
        "unfold_input_into_patches_f64",
        "unfold_output_into_patches_f64",
        "transpose_filters_f64",
    ];
}

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => unreachable!("Only implemented for 3d & 4d arrays"),
    }
}

impl<E: Dtype + ValidAsZeroBits> super::ConvTrans2DKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
    CudaBlas: Gemm<E>,
{
    fn alloc<S: Shape>(&self, shape: S) -> Result<Tensor<S, E, Self>, Self::Err> {
        let data = unsafe { self.alloc_empty::<E>(shape.num_elements()) }?;
        Ok(self.build_tensor(shape, shape.strides(), data))
    }

    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let patches_numel = op.batch * op.chan_in * op.kernel * op.kernel * op.h_out * op.w_out;
        let mut patches = unsafe { self.get_workspace::<E>(patches_numel) }?;
        let mut patches = unsafe { patches.transmute_mut::<E>(patches_numel).unwrap() };

        let ftr_numel = op.groups
            * (op.chan_out / op.groups)
            * (op.chan_in / op.groups)
            * op.kernel
            * op.kernel;
        let mut ftr = unsafe { self.alloc_empty::<E>(ftr_numel) }?;

        let img_strides = self.dev.htod_copy(make_4d::<L>(lhs.strides).into())?;
        let f_strides = self.dev.htod_copy(rhs.strides.into())?;

        let out_buf = Arc::get_mut(&mut out.data).unwrap();

        // LHS    (G, O/G, C/G*K*K)
        // RHS (B, G, C/G*K*K, OH*OW)
        // OUT (B, G, O/G, OH*OW)
        let m = op.chan_out / op.groups;
        let k = (op.chan_in / op.groups) * op.kernel * op.kernel;
        let n = op.h_out * op.w_out;
        unsafe {
            // generate patches for matmul
            let unfold_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
            let cfg = launch_cfg::<128>((op.batch * op.chan_in * op.h_out * op.w_out) as u32);
            unfold_fn.launch(cfg, (op, lhs.data.as_ref(), &img_strides, &mut patches))?;

            // prepare filters for backward operations by
            // swapping dims 0 and 1 and adding a batch dimension
            let tr_fn = self.dev.get_func(Self::MOD, Self::FNS[2]).unwrap();
            let cfg = launch_cfg::<128>(rhs.shape.num_elements() as u32);
            tr_fn.launch(cfg, (op, rhs.data.as_ref(), &f_strides, &mut ftr))?;

            if op.groups == 1 {
                self.gemm_batch(
                    (op.batch, m, k, n),
                    &ftr,
                    [0, k, 1],
                    &patches,
                    [k * n, n, 1],
                    Default::default(),
                    out_buf,
                    [m * n, n, 1],
                )
                .unwrap();
            } else {
                for i_batch in 0..op.batch {
                    self.gemm_batch(
                        (op.groups, m, k, n),
                        &ftr,
                        [m * k, k, 1],
                        &patches.slice(i_batch * op.groups * k * n..),
                        [k * n, n, 1],
                        Default::default(),
                        &mut out_buf.slice_mut(i_batch * op.groups * m * n..),
                        [m * n, n, 1],
                    )
                    .unwrap();
                }
            }
        }

        Ok(())
    }

    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        _: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let patches_numel = op.batch * op.chan_out * op.kernel * op.kernel * op.h_in * op.w_in;

        let mut patches = unsafe { self.get_workspace::<E>(patches_numel) }?;
        let mut patches = unsafe { patches.transmute_mut::<E>(patches_numel).unwrap() };

        {
            // unfold grad_out into patches
            let unfold_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
            let cfg = launch_cfg::<128>((op.batch * op.chan_out * op.h_in * op.w_in) as u32);
            unsafe { unfold_fn.launch(cfg, (op, grad_out, &mut patches)) }?;
        }

        let rhs_buf = rhs.data.as_ref();
        let lhs_buf = lhs.data.as_ref();

        unsafe {
            self.par_stream.wait_for_default()?;

            // img_g += filters * patches
            // LHS =    (G, C/G, O/G*K*K)
            // RHS = (B, G, O/G*K*K, H*W)
            // OUT = (B, G, C/G, H*W)
            let m = op.chan_in / op.groups;
            let k = (op.chan_out / op.groups) * op.kernel * op.kernel;
            let n = op.h_in * op.w_in;
            self.blas.set_stream(Some(self.par_stream.as_ref()))?;
            if op.groups == 1 {
                // optimizing here for common case
                self.gemm_batch(
                    (op.batch, m, k, n),
                    rhs_buf,
                    [0, k, 1],
                    &patches,
                    [k * n, n, 1],
                    <E>::ONE,
                    grad_lhs,
                    [m * n, n, 1],
                )
                .unwrap();
            } else {
                for i_batch in 0..op.batch {
                    self.gemm_batch(
                        (op.groups, m, k, n),
                        rhs_buf,
                        [m * k, k, 1],
                        &patches.slice(i_batch * op.groups * k * n..),
                        [k * n, n, 1],
                        <E>::ONE,
                        &mut grad_lhs.slice_mut(i_batch * op.groups * m * n..),
                        [m * n, n, 1],
                    )
                    .unwrap();
                }
            }
            self.blas.set_stream(None)?;
        }

        unsafe {
            // weight_g += img * patches^T
            // LHS = (B, G, C/G, H*W)
            // RHS = (B, H*W, G, O/G*K*K)
            // OUT =    (G, C/G, O/G*K*K)
            let m = op.chan_in / op.groups;
            let k = op.h_in * op.w_in;
            let n = (op.chan_out / op.groups) * op.kernel * op.kernel;
            if op.groups == 1 {
                // optimizing here for common case
                for i_batch in 0..op.batch {
                    self.gemm(
                        (m, k, n),
                        &lhs_buf.slice(i_batch * m * k..),
                        [k, 1],
                        &patches.slice(i_batch * k * n..),
                        [1, k],
                        E::ONE,
                        grad_rhs,
                        [n, 1],
                    )
                    .unwrap()
                }
            } else {
                for i_batch in 0..op.batch {
                    self.gemm_batch(
                        (op.groups, m, k, n),
                        &lhs_buf.slice(i_batch * op.groups * m * k..),
                        [m * k, k, 1],
                        &patches.slice(i_batch * op.groups * k * n..),
                        [k * n, 1, k],
                        E::ONE,
                        grad_rhs,
                        [m * n, n, 1],
                    )
                    .unwrap();
                }
            }
        }

        self.dev.wait_for(self.par_stream.as_ref())?;

        Ok(())
    }
}
