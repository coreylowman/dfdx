use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
};
use cudarc::driver::{DeviceSlice, LaunchAsync};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/reshape.ptx"));

trait HasCudaKernel<E> {
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
    const MOD: &'static str = "reshape_f32";
    const FNS: &'static [&'static str] = &["reshape_fwd_f32", "reshape_bwd_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "reshape_f64";
    const FNS: &'static [&'static str] = &["reshape_fwd_f64", "reshape_bwd_f64"];
}

impl<E: Dtype> super::ReshapeKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: &Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let numel = inp.data.len();
        let mut storage = unsafe { self.dev.alloc::<E>(numel) }?;

        let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
        let dst_dims = self.dev.htod_copy(dst.concrete().into())?;
        let inp_strides = self.dev.htod_copy(inp.strides.into())?;
        let dst_strides = self.dev.htod_copy(dst.strides().into())?;

        let fwd_fn = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
        let cfg = launch_cfg(numel as u32);
        let params = (
            numel,             // const size_t numel,
            inp.data.as_ref(), // const float *inp,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            &inp_dims,         // const size_t *inp_dims,
            &inp_strides,      // const size_t *inp_strides,
            &mut storage,      // float *out
            Dst::NUM_DIMS,     // const size_t out_num_dims,
            &dst_dims,         // const size_t *out_dims,
            &dst_strides,      // const size_t *out_strides,
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(self.build_tensor(*dst, dst.strides(), storage))
    }

    fn backward<Src: Shape, Dst: Shape>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<Dst, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();
        let numel = grad_inp.len();

        let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
        let out_dims = self.dev.htod_copy(out.shape.concrete().into())?;
        let inp_strides = self.dev.htod_copy(inp.strides.into())?;
        let out_strides = self.dev.htod_copy(out.strides.into())?;

        let cfg = launch_cfg(numel as u32);
        let params = (
            numel,         // const size_t numel,
            grad_inp,      // float *grad_inp,
            Src::NUM_DIMS, // const size_t inp_num_dims,
            &inp_dims,     // const size_t *inp_dims,
            &inp_strides,  // const size_t *inp_strides,
            grad_out,      // const float *grad_out,
            Dst::NUM_DIMS, // const size_t out_num_dims,
            &out_dims,     // const size_t *out_dims,
            &out_strides,  // const size_t *out_strides
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
