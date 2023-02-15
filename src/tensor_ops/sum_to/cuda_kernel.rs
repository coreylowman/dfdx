use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
    tensor_ops::reduction_utils::*,
};

use cudarc::driver::{AsKernelParam, CudaSlice, LaunchAsync, LaunchConfig, ValidAsZeroBits};

use std::sync::Arc;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/sum_to.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "sum_f32";
    const FNS: &'static [&'static str] = &["sum_to_fwd_f32", "sum_to_bwd_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "sum_f64";
    const FNS: &'static [&'static str] = &["sum_to_fwd_f64", "sum_to_bwd_f64"];
}

impl<E: Dtype + ValidAsZeroBits + AsKernelParam> super::SumKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();

        let (dims, strides) = permute_for_reductions::<_, Ax>(inp.shape.concrete(), inp.strides);
        let num_dims = dims.len();
        let dims: CudaSlice<usize> = self.dev.take_async(dims)?;
        let strides: CudaSlice<usize> = self.dev.take_async(strides)?;

        let mut storage = self.dev.alloc_zeros_async::<E>(dst.num_elements())?;

        let elems_per_thread = E::from_usize(reduction_elems_per_thread::<Ax, Src>(
            inp.shape.concrete(),
            inp.strides,
        ))
        .unwrap();

        let physical_numel = inp.data.len();
        let (dst_physical_numel, dst_strides) =
            reduction_output_strides::<Ax, Src, Dst>(inp.strides, dst);
        let chunk_len = physical_numel / dst_physical_numel;

        let cfg = LaunchConfig::for_num_elems(physical_numel as u32);
        let params = (
            physical_numel,    // const size_t numel,
            num_dims,          // const size_t num_dims,
            elems_per_thread,  // const float elems_per_thread,
            chunk_len,         // const size_t chunk_len,
            inp.data.as_ref(), // const float *inp,
            &dims,             // const size_t *dims,
            &strides,          // const size_t *strides,
            &mut storage,      // float *out
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(CudaArray {
            data: Arc::new(storage),
            shape: dst,
            strides: dst_strides,
        })
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();

        let dims: CudaSlice<usize> = self.dev.take_async(grad_inp.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(grad_inp.strides.into())?;
        let out_strides: Src::Concrete =
            BroadcastStridesTo::<Src, Ax>::broadcast_strides(&grad_out.shape, grad_out.strides);
        let out_strides: CudaSlice<usize> = self.dev.take_async(out_strides.into())?;

        let physical_numel = grad_inp.data.len();
        let elems_per_thread = E::from_usize(reduction_elems_per_thread::<Ax, Src>(
            grad_inp.shape.concrete(),
            grad_inp.strides,
        ))
        .unwrap();

        let cfg = LaunchConfig::for_num_elems(physical_numel as u32);
        let params = (
            physical_numel,                    // const size_t numel,
            Src::NUM_DIMS,                     // const size_t num_dims,
            elems_per_thread,                  // const float elems_per_thread,
            &dims,                             // const size_t *dims,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            &inp_strides,                      // const size_t *inp_strides,
            grad_out.data.as_ref(),            // const float *grad_out,
            &out_strides,                      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}
