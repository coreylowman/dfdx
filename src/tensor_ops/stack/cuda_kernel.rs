use crate::{
    shapes::*,
    tensor::cuda::{Cuda, CudaArray},
};
use cudarc::driver::{result::memcpy_dtod_async, DevicePtr, DevicePtrMut};
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
        inp: Vec<&Self::Storage<S, E>>,
    ) -> Result<Self::Storage<S::Larger, E>, Self::Err>
    where
        S: super::AddDim<Num>,
    {
        debug_assert_eq!(inp.len(), num.size());

        // check that all the strides are the same
        let item_strides = inp[0].strides;
        for i in inp.iter() {
            assert_eq!(i.strides, item_strides);
        }
        let shape: S::Larger = inp[0].shape().add(num);

        // build the new strides
        let mut strides = shape.strides();
        strides[0] = inp[0].data.len();
        for d in 1..<S::Larger as Shape>::NUM_DIMS {
            strides[d] = item_strides[d - 1];
        }

        // copy the data
        let numel = strides[0];
        let mut data = unsafe { self.dev.alloc_async::<E>(numel) }?;
        let mut offset = 0;
        for i in inp {
            let num = i.data.len();
            let num_bytes = num * std::mem::size_of::<E>();
            let s = data.try_slice_mut(offset..offset + num).unwrap();
            // unsafe {
            //     memcpy_dtod_async(
            //         *s.device_ptr_mut(),
            //         *i.data.device_ptr(),
            //         num_bytes,
            //         self.dev.stream,
            // }?;
            offset += num;
        }
        todo!()

        // Ok(StridedArray {
        //     data: std::sync::Arc::new(data),
        //     shape,
        //     strides,
        // })
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
            let params = (
                numel,
                &grad_out_buf.try_slice(offset..offset + numel).unwrap(),
                Arc::make_mut(&mut item.data),
            );
            unsafe { f.launch_async(cfg, params) }?;
            offset += numel;
        }
        Ok(())
    }
}
