use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
};
use cudarc::driver::{DeviceSlice, LaunchAsync};
use std::vec::Vec;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/stack.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "stack_f32";
    const FNS: &'static [&'static str] = &["sum_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "stack_f64";
    const FNS: &'static [&'static str] = &["sum_f64"];
}

impl<E: Dtype> super::StackKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inps: &[Tensor<S, E, Self>],
    ) -> Result<Tensor<S::Larger, E, Self>, Self::Err>
    where
        S: super::AddDim<Num>,
    {
        debug_assert_eq!(inps.len(), num.size());

        // check that all the strides are the same
        let item_strides = inps[0].strides;
        for i in inps.iter() {
            assert_eq!(i.strides, item_strides);
        }
        let shape: S::Larger = inps[0].shape().add_dim(num);

        // build the new strides
        let mut strides = shape.strides();
        strides[0] = inps[0].data.len();
        for d in 1..<S::Larger as Shape>::NUM_DIMS {
            strides[d] = item_strides[d - 1];
        }

        // copy the data
        let item_numel = strides[0];
        let mut data = unsafe { self.dev.alloc::<E>(num.size() * item_numel) }?;
        let mut offset = 0;
        for item in inps {
            debug_assert_eq!(item.data.len(), item_numel);
            self.dev.dtod_copy(
                item.data.as_ref(),
                &mut data.slice_mut(offset..offset + item_numel),
            )?;
            offset += item_numel;
        }
        debug_assert_eq!(offset, data.len());
        Ok(self.build_tensor(shape, strides, data))
    }

    fn backward(
        &self,
        mut grad_inp: Vec<&mut Self::Vec<E>>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX.into(), Self::MOD, Self::FNS)?;
        }
        let mut offset = 0;
        for item in grad_inp.drain(..) {
            let f = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
            let numel: usize = item.len();
            let cfg = launch_cfg(numel as u32);
            let sub = grad_out.slice(offset..offset + numel);
            unsafe { f.launch(cfg, (numel, &sub, item)) }?;
            offset += numel;
        }
        debug_assert_eq!(offset, grad_out.len());
        Ok(())
    }
}
