use crate::{
    shapes::*,
    tensor::{launch_cfg, unique_id, Cuda, Tensor},
};
use cudarc::driver::{DeviceSlice, LaunchAsync};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/concat.ptx"));

trait HasCudaKernel<E> {
    const BWD_FN: &'static str;
}
impl HasCudaKernel<f32> for Cuda {
    const BWD_FN: &'static str = "concat_bwd_f32";
}
impl HasCudaKernel<f64> for Cuda {
    const BWD_FN: &'static str = "concat_bwd_f64";
}

impl<E: Dtype> super::ConcatKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<A: Shape, B: Shape>(
        &self,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
    ) -> Result<Tensor<A::Catted, E, Self>, Self::Err>
    where
        A: super::ConcatShape<B>,
    {
        debug_assert_eq!(a.strides, a.shape.strides());
        debug_assert_eq!(b.strides, b.shape.strides());
        let shape = a.shape.concat_shape(&b.shape);
        let mut buf = unsafe { self.dev.alloc::<E>(shape.num_elements()) }?;
        debug_assert_eq!(buf.len(), a.data.len() + b.data.len());
        self.dev
            .dtod_copy(a.data.as_ref(), &mut buf.slice_mut(0..a.data.len()))?;
        self.dev
            .dtod_copy(b.data.as_ref(), &mut buf.slice_mut(a.data.len()..))?;
        Ok(Tensor {
            id: unique_id(),
            data: std::sync::Arc::new(buf),
            shape,
            strides: shape.strides(),
            device: self.clone(),
            tape: Default::default(),
        })
    }
    fn backward<A: Shape, B: Shape>(
        &self,
        _: &Tensor<A, E, Self>,
        grad_a: &mut Self::Vec<E>,
        _: &Tensor<B, E, Self>,
        grad_b: &mut Self::Vec<E>,
        _: &Tensor<A::Catted, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        A: super::ConcatShape<B>,
    {
        if !self.dev.has_func(Self::BWD_FN, Self::BWD_FN) {
            self.dev
                .load_ptx(PTX.into(), Self::BWD_FN, &[Self::BWD_FN])?;
        }

        let mut offset = 0;
        {
            let f = self.dev.get_func(Self::BWD_FN, Self::BWD_FN).unwrap();
            let numel = grad_a.len();
            let cfg = launch_cfg(numel as u32);
            unsafe { f.launch(cfg, (numel, &grad_out.slice(0..numel), grad_a)) }?;
            offset += numel;
        }
        {
            let f = self.dev.get_func(Self::BWD_FN, Self::BWD_FN).unwrap();
            let numel = grad_b.len();
            let cfg = launch_cfg(numel as u32);
            unsafe { f.launch(cfg, (numel, &grad_out.slice(offset..), grad_b)) }?;
        }
        Ok(())
    }
}
