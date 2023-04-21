use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor, Tensorlike},
    tensor_ops::reduction_utils::*,
};

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, ValidAsZeroBits};

use std::vec::Vec;

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

impl<E: Dtype + ValidAsZeroBits + DeviceRepr> super::SumKernel<E> for Cuda
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

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();

        let (dims, strides) = permute_for_reductions::<_, Ax>(inp.shape.concrete(), inp.strides);
        let num_dims = dims.len();

        let mut info = Vec::with_capacity(num_dims * 2);
        info.extend(dims);
        info.extend(strides);
        let info = self.dev.htod_copy(info)?;

        let elems_per_thread = E::from_usize(reduction_elems_per_thread::<_, Src>(
            inp.shape.concrete(),
            inp.strides,
            Ax::as_array(),
        ))
        .unwrap();

        let physical_numel = inp.data.len();
        let (dst_physical_numel, dst_strides) =
            reduction_output_strides::<Ax, Src, Dst>(inp.strides, dst);
        let chunk_len = physical_numel / dst_physical_numel;

        let cfg = launch_cfg::<128>(physical_numel as u32);

        let mut storage = unsafe { self.alloc_empty::<E>(dst_physical_numel) }?;
        self.dev.memset_zeros(&mut storage)?;
        let params = (
            physical_numel,    // const size_t numel,
            num_dims,          // const size_t num_dims,
            elems_per_thread,  // const float elems_per_thread,
            chunk_len,         // const size_t chunk_len,
            &info,             // const size_t *info,
            inp.data.as_ref(), // const float *inp,
            &mut storage,      // float *out
        );
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(self.build_tensor(dst, dst_strides, storage))
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &impl Tensorlike<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();

        let out_strides: Src::Concrete =
            BroadcastStridesTo::<Src, Ax>::broadcast_strides(&dst, dst.strides());
        let physical_numel = inp.len();
        let elems_per_thread = E::from_usize(reduction_elems_per_thread::<_, Src>(
            inp.shape().concrete(),
            inp.strides(),
            Ax::as_array(),
        ))
        .unwrap();

        let cfg = launch_cfg::<128>(physical_numel as u32);

        let mut info: Vec<usize> = Vec::with_capacity(3 * Src::NUM_DIMS);
        info.extend(inp.shape().concrete());
        info.extend(inp.strides());
        info.extend(out_strides);
        let info = self.dev.htod_copy(info)?;

        let params = (
            physical_numel,   // const size_t numel,
            Src::NUM_DIMS,    // const size_t num_dims,
            elems_per_thread, // const float elems_per_thread,
            &info,            // const size_t *info,
            grad_inp,         // float *grad_inp,
            grad_out,         // const float *grad_out,
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
