use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::driver::{DeviceRepr, LaunchAsync, ValidAsZeroBits};

use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor, Tensorlike},
};

use std::sync::Arc;

unsafe impl DeviceRepr for super::Conv2DOp {}

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/conv2d.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "conv2d_f32";
    const FNS: &'static [&'static str] = &[
        "unfold_input_into_patches_f32",
        "unfold_output_into_patches_f32",
        "transpose_filters_f32",
        "sum_transposed_filters_f32",
    ];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "conv2d_f64";
    const FNS: &'static [&'static str] = &[
        "unfold_input_into_patches_f64",
        "unfold_output_into_patches_f64",
        "transpose_filters_f64",
        "sum_transposed_filters_f64",
    ];
}

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => unreachable!("Only implemented for 3d & 4d arrays"),
    }
}

impl<E: Dtype + ValidAsZeroBits> super::Conv2DKernel<E> for Cuda
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
        op: super::Conv2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let patches_item_numel = op.chan_in * op.kernel * op.kernel * op.h_out * op.w_out;
        let patches_numel = op.batch * patches_item_numel;

        let mut patches = unsafe { self.get_workspace::<E>(patches_numel) }?;
        let mut patches = unsafe { patches.transmute_mut::<E>(patches_numel).unwrap() };

        let img_strides = self.dev.htod_copy(make_4d::<L>(lhs.strides).into())?;
        let unfold_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
        let cfg = launch_cfg::<128>((op.batch * op.chan_in * op.h_out * op.w_out) as u32);
        let params = (op, lhs.data.as_ref(), &img_strides, &mut patches);
        unsafe { unfold_fn.launch(cfg, params) }?;

        // (O, C * K * K) * (B, C * K * K, OH * OW) = (B, O, OH * OW)
        let m = op.chan_out;
        let k = op.chan_in * op.kernel * op.kernel;
        let n = op.h_out * op.w_out;
        unsafe {
            self.gemm_batch(
                (op.batch, m, k, n),
                rhs.data.as_ref(),
                [0, k, 1],
                &patches,
                [k * n, n, 1],
                Default::default(),
                Arc::get_mut(&mut out.data).unwrap(),
                [m * n, n, 1],
            )
            .unwrap();
        }

        Ok(())
    }

    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::Conv2DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        _: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let patches_item_numel = op.chan_out * op.kernel * op.kernel * op.h_in * op.w_in;
        let patches_numel = op.batch * patches_item_numel;
        let filters_numel = op.chan_in * op.chan_out * op.kernel * op.kernel;

        let mut patches = unsafe { self.get_workspace::<E>(patches_numel) }?;
        let mut patches = unsafe { patches.transmute_mut::<E>(patches_numel).unwrap() };

        let mut f_b1023 = unsafe { self.alloc_empty::<E>(filters_numel) }?;
        let mut grad_f_b1023 = unsafe { self.alloc_empty::<E>(op.batch * filters_numel) }?;
        let f_strides = self.dev.htod_copy(rhs.strides.into())?;

        self.par_stream.wait_for_default()?;

        {
            // unfold grad_out into patches
            let unfold_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
            let cfg = launch_cfg::<128>((op.batch * op.chan_out * op.h_in * op.w_in) as u32);
            unsafe { unfold_fn.launch(cfg, (op, grad_out, &mut patches)) }?;
        }

        {
            // prepare filters for backward operations by
            // swapping dims 0 and 1
            let tr_fn = self.dev.get_func(Self::MOD, Self::FNS[2]).unwrap();
            let cfg = launch_cfg::<128>(rhs.shape.num_elements() as u32);
            unsafe {
                tr_fn.launch_on_stream(
                    self.par_stream.as_ref(),
                    cfg,
                    (op, rhs.data.as_ref(), &f_strides, &mut f_b1023),
                )
            }?;

            self.par_stream.wait_for_default()?;

            // img_g += filters * patches
            // (B, C, H * W) += (B, C, O * K * K) * (B, O * K * K, H * W)
            let m = op.chan_in;
            let k = op.chan_out * op.kernel * op.kernel;
            let n = op.h_in * op.w_in;
            unsafe {
                self.blas.set_stream(Some(self.par_stream.as_ref()))?;
                self.gemm_batch(
                    (op.batch, m, k, n),
                    &f_b1023,
                    [0, k, 1],
                    &patches,
                    [k * n, n, 1],
                    <E>::ONE,
                    grad_lhs,
                    [m * n, n, 1],
                )
                .unwrap();
                self.blas.set_stream(None)?;
            }
        }

        {
            // weight_g += img * patches^T
            // (B, C, O * K * K) += (B, C, H * W) * (B, H * W, O * K * K)
            let m = op.chan_in;
            let k = op.h_in * op.w_in;
            let n = op.chan_out * op.kernel * op.kernel;
            unsafe {
                self.gemm_batch(
                    (op.batch, m, k, n),
                    lhs.data.as_ref(),
                    [m * k, k, 1],
                    &patches,
                    [k * n, 1, k],
                    Default::default(),
                    &mut grad_f_b1023,
                    [m * n, n, 1],
                )
                .unwrap();
            }

            // sum all the gradients collected in our broadcasted grad_f
            // into grad_rhs
            let sum_fn = self.dev.get_func(Self::MOD, Self::FNS[3]).unwrap();
            let cfg = launch_cfg::<128>(rhs.shape.num_elements() as u32);
            unsafe { sum_fn.launch(cfg, (op, &grad_f_b1023, grad_rhs, &f_strides)) }?;
        }

        self.dev.wait_for(self.par_stream.as_ref())?;

        Ok(())
    }
}
