use crate::{
    prelude::cpu::NdIndex,
    shapes::*,
    tensor::{Cuda, Tensor},
};
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/slice.ptx"));

pub(crate) trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

macro_rules! has_kernels {
    ($($dtype:ty),*) => {
        $(
        impl HasCudaKernel<$dtype> for Cuda {
            const MOD: &'static str = concat!("slice_", stringify!($dtype));
            const FNS: &'static [&'static str] = &[concat!("slice_fwd_", stringify!($dtype))];
        }
        )*
    }
}

has_kernels!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, bool);

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "slice_f32";
    const FNS: &'static [&'static str] = &["slice_fwd_f32", "slice_bwd_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "slice_f64";
    const FNS: &'static [&'static str] = &["slice_fwd_f64", "slice_bwd_f64"];
}

impl<E: Dtype> super::SliceKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape + SliceShape<Slice>, Slice>(
        &self,
        inp: &Tensor<Src, E, Self>,
        slice: &Slice,
    ) -> Result<Tensor<Src::Sliced, E, Self>, Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let dst = inp.shape.slice(slice).unwrap();
        let strides = inp.strides;
        let numel = dst.num_elements();

        let start_idx = NdIndex::new(inp.shape, inp.strides)
            .get_strided_index(inp.shape.first_idx_in_slice(slice));

        let mut storage = unsafe { self.dev.alloc::<E>(numel) }?;

        let dims: CudaSlice<usize> = self.dev.htod_copy(dst.concrete().into())?;
        let strides: CudaSlice<usize> = self.dev.htod_copy(strides.into())?;

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            Src::NUM_DIMS,     // const size_t num_dims,
            &dims,             // const size_t *dims,
            &strides,          // const size_t *strides,
            start_idx,         // const size_t offset,
            inp.data.as_ref(), // const T *inp,
            &mut storage,      // T *out
        );
        unsafe { fwd_fn.launch(cfg, params) }?;
        Ok(self.build_tensor(dst, dst.strides(), storage))
    }

    fn backward<Src: Shape + SliceShape<Slice>, Slice>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut CudaSlice<E>,
        grad_out: &CudaSlice<E>,
        slice: &Slice,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FNS[1]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let dst = inp.shape.slice(slice).unwrap();
        let strides = inp.strides;
        let numel = dst.num_elements();

        let start_idx = NdIndex::new(inp.shape, inp.strides)
            .get_strided_index(inp.shape.first_idx_in_slice(slice));

        let dims: CudaSlice<usize> = self.dev.htod_copy(dst.concrete().into())?;
        let strides: CudaSlice<usize> = self.dev.htod_copy(strides.into())?;

        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,         // const size_t numel,
            Src::NUM_DIMS, // const size_t num_dims,
            &dims,         // const size_t *dims,
            &strides,      // const size_t *strides,
            start_idx,     // const size_t offset,
            grad_inp,      // T *grad_inp,
            grad_out,      // const T *out
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
