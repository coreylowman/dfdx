use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
    tensor_ops::reduction_utils::*,
};

use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/max_to.ptx"));

trait HasCudaKernel<E> {
    const INIT: E;
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const INIT: f32 = f32::NEG_INFINITY;
    const MOD: &'static str = "max_f32";
    const FNS: &'static [&'static str] = &["max_to_fwd_f32", "max_to_bwd_f32", "fill_with_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const INIT: f64 = f64::NEG_INFINITY;
    const MOD: &'static str = "max_f64";
    const FNS: &'static [&'static str] = &["max_to_fwd_f64", "max_to_bwd_f64", "fill_with_f64"];
}

impl<E: Dtype> super::MaxReduceKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let fill_fn = self.dev.get_func(Self::MOD, Self::FNS[2]).unwrap();
        let mut storage = unsafe {
            let mut storage = self.dev.alloc::<E>(dst.num_elements())?;
            fill_fn.launch(
                launch_cfg(dst.num_elements() as u32),
                (&mut storage, Self::INIT, dst.num_elements()),
            )?;
            storage
        };

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();

        let (dims, strides) = permute_for_reductions::<_, Ax>(inp.shape.concrete(), inp.strides);
        let dims: CudaSlice<usize> = self.dev.htod_copy(dims)?;
        let strides: CudaSlice<usize> = self.dev.htod_copy(strides)?;

        let physical_numel = inp.data.len();
        let (dst_physical_numel, dst_strides) =
            reduction_output_strides::<Ax, Src, Dst>(inp.strides, dst);
        let chunk_len = physical_numel / dst_physical_numel;

        let cfg = launch_cfg(physical_numel as u32);
        let params = (
            physical_numel,    // const size_t numel,
            dims.len(),        // const size_t num_dims,
            chunk_len,         // const size_t chunk_len,
            inp.data.as_ref(), // const float *inp,
            &dims,             // const size_t *dims,
            &strides,          // const size_t *strides,
            &mut storage,      // float *out
        );
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(self.build_tensor(dst, dst_strides, storage))
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();

        let dims: CudaSlice<usize> = self.dev.htod_copy(inp.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.htod_copy(inp.strides.into())?;
        let out_strides: Src::Concrete =
            BroadcastStridesTo::<Src, Ax>::broadcast_strides(&out.shape, out.strides);
        let out_strides: CudaSlice<usize> = self.dev.htod_copy(out_strides.into())?;

        let physical_numel = grad_inp.len();
        let elems_per_thread = E::from_usize(reduction_elems_per_thread::<_, Src>(
            inp.shape.concrete(),
            inp.strides,
            Ax::as_array(),
        ))
        .unwrap();

        let cfg = launch_cfg(physical_numel as u32);
        let params = (
            physical_numel,    // const size_t numel,
            Src::NUM_DIMS,     // const size_t num_dims,
            elems_per_thread,  // const float elems_per_thread,
            &dims,             // const size_t *dims,
            inp.data.as_ref(), // const float *inp,
            grad_inp,          // float *grad_inp,
            &inp_strides,      // const size_t *inp_strides,
            out.data.as_ref(), // const float *out,
            grad_out,          // const float *grad_out,
            &out_strides,      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
