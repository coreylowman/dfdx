use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::{sync::Arc, vec::Vec};

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
        inps: Vec<&Self::Storage<S, E>>,
    ) -> Result<Self::Storage<S::Larger, E>, Self::Err>
    where
        S: super::AddDim<Num>,
    {
        debug_assert_eq!(inps.len(), num.size());

        // check that all the strides are the same
        let item_strides = inps[0].strides;
        for i in inps.iter() {
            assert_eq!(i.strides, item_strides);
        }
        let shape: S::Larger = inps[0].shape().add(num);

        // build the new strides
        let mut strides = shape.strides();
        strides[0] = inps[0].data.len();
        for d in 1..<S::Larger as Shape>::NUM_DIMS {
            strides[d] = item_strides[d - 1];
        }

        // copy the data
        let item_numel = strides[0];
        let mut data = unsafe { self.dev.alloc_async::<E>(num.size() * item_numel) }?;
        let mut offset = 0;
        for item in inps {
            debug_assert_eq!(item.data.len(), item_numel);
            self.dev.device_copy_async(
                item.data.as_ref(),
                &mut data.try_slice_mut(offset..offset + item_numel).unwrap(),
            )?;
            offset += item_numel;
        }
        debug_assert_eq!(offset, data.len());
        Ok(CudaArray {
            data: std::sync::Arc::new(data),
            shape,
            strides,
        })
    }
    fn backward<S: Shape, New: Dim>(
        &self,
        mut grad_inp: Vec<&mut Self::Storage<S, E>>,
        grad_out: &Self::Storage<S::Larger, E>,
    ) -> Result<(), Self::Err>
    where
        S: super::AddDim<New>,
    {
        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX.into(), Self::MOD, Self::FNS)?;
        }
        let grad_out_buf = grad_out.data.as_ref();
        let mut offset = 0;
        for item in grad_inp.drain(..) {
            let f = self.dev.get_func(Self::MOD, Self::FNS[0]).unwrap();
            let numel: usize = item.data.len();
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let sub = grad_out_buf.try_slice(offset..offset + numel).unwrap();
            let params = (numel, &sub, Arc::make_mut(&mut item.data));
            unsafe { f.launch_async(cfg, params) }?;
            offset += numel;
        }
        debug_assert_eq!(offset, grad_out_buf.len());
        Ok(())
    }
}
