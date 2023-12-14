#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::*,
    tensor::{masks::triangle_mask, storage_traits::*, unique_id, Cpu, Error, NoneTape, Tensor},
};

use super::{device::CachableBuffer, Buffer, Webgpu};

use core::marker::PhantomData;
use rand::Rng;
use std::{sync::Arc, vec::Vec};
use wgpu::COPY_BUFFER_ALIGNMENT;

pub(crate) fn round_to_buffer_alignment(size: u64) -> u64 {
    (size + (COPY_BUFFER_ALIGNMENT - 1)) / COPY_BUFFER_ALIGNMENT * COPY_BUFFER_ALIGNMENT
}

impl Webgpu {
    fn tensor_from_host_buf<S: Shape, E: Unit>(
        &self,
        shape: S,
        buf: Vec<E>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let buffer = self.alloc_empty::<E>(buf.len())?;
        buffer.copy_to_device::<E>(&self.dev, &self.queue, &buf);

        Ok(self.build_tensor(shape, shape.strides(), buffer))
    }

    pub(crate) fn build_tensor<S: Shape, E: Unit>(
        &self,
        shape: S,
        strides: S::Concrete,
        buffer: Buffer,
    ) -> Tensor<S, E, Self> {
        let data = CachableBuffer {
            dev: self.dev.clone(),
            queue: self.queue.clone(),
            data: buffer,
            cache: self.cache.clone(),
            _phantom: PhantomData,
        };
        Tensor {
            id: unique_id(),
            data: Arc::new(data),
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        }
    }
}

impl<E: Unit + SafeZeros> ZerosTensor<E> for Webgpu {
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let strides = shape.strides();
        let data = self.alloc_empty::<E>(shape.num_elements())?;
        data.copy_to_device(&self.dev, &self.queue, &vec![0u8; data.size()]);

        Ok(self.build_tensor(shape, strides, data))
    }
}

impl<E: Unit + SafeZeros> ZeroFillStorage<E> for Webgpu {
    fn try_fill_with_zeros(&self, storage: &mut Self::Vec) -> Result<(), Error> {
        storage.copy_to_device(&self.dev, &self.queue, &vec![0u8; storage.size()]);

        Ok(())
    }
}

impl<E: Unit> WithStorage<E> for Webgpu {
    fn try_element_view<F: FnMut(&E)>(&self, storage: &Self::Vec, mut f: F) -> Result<(), Error> {
        todo!()
    }
    fn try_view<F: FnMut(&[E])>(&self, storage: &Self::Vec, mut f: F) -> Result<(), Error> {
        todo!()
    }
    fn try_element_map<F: FnMut(E) -> E>(
        &self,
        storage: &mut Self::Vec,
        mut f: F,
    ) -> Result<(), Error> {
        todo!()
    }
    fn try_map<F: FnMut(Vec<E>) -> Option<Vec<E>>>(
        &self,
        storage: &mut Self::Vec,
        mut f: F,
    ) -> Result<(), Error> {
        todo!()
    }
}

impl<E: Unit> OnesTensor<E> for Webgpu {
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let buf = vec![E::ONE; shape.num_elements()];
        self.tensor_from_host_buf(shape, buf)
    }
}

impl<E: Unit> TriangleTensor<E> for Webgpu
where
    Cpu: TriangleTensor<E>,
{
    fn try_upper_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let mut data = vec![val; shape.num_elements()];
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(&mut data, &shape, true, offset);
        self.tensor_from_host_buf(shape, data)
    }

    fn try_lower_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let mut data = vec![val; shape.num_elements()];
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(&mut data, &shape, false, offset);
        self.tensor_from_host_buf(shape, data)
    }
}

impl<E: Unit> OneFillStorage<E> for Webgpu {
    fn try_fill_with_ones(&self, storage: &mut Self::Vec) -> Result<(), Error> {
        let len = storage.size() as usize / std::mem::size_of::<E>();
        let buf = vec![E::ONE; len];
        storage
            .data
            .copy_to_device::<E>(&self.dev, &self.queue, &buf);

        Ok(())
    }
}

impl<E: Unit> SampleTensor<E> for Webgpu
where
    Cpu: SampleTensor<E>,
{
    fn try_sample_like<S: HasShape, D: rand::prelude::Distribution<E>>(
        &self,
        src: &S,
        distr: D,
    ) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let mut buf = Vec::with_capacity(shape.num_elements());
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.cpu.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.cpu.rng.lock();
            buf.resize_with(shape.num_elements(), || rng.sample(&distr));
        }
        self.tensor_from_host_buf::<S::Shape, E>(shape, buf)
    }

    fn try_fill_with_distr<D: rand::prelude::Distribution<E>>(
        &self,
        storage: &mut Self::Vec,
        distr: D,
    ) -> Result<(), Error> {
        let len = storage.size() as usize / std::mem::size_of::<E>();
        let mut buf = Vec::with_capacity(len);
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.cpu.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.cpu.rng.lock();
            buf.resize_with(len, || rng.sample(&distr));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                storage.data.slice(..).get_mapped_range_mut().as_mut_ptr() as *mut E,
                len,
            )
        };
        Ok(())
    }
}

impl<E: Unit> CopySlice<E> for Webgpu {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]) {
        assert_eq!(
            dst.data.size() as usize,
            src.len() * std::mem::size_of::<E>(),
            "Slices must have same number of elements as *physical* Storage<E> of tensors."
        );
        dst.data
            .data
            .copy_to_device(&dst.device.dev, &dst.device.queue, src);
    }

    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]) {
        assert_eq!(
            src.data.size() as usize,
            dst.len() * std::mem::size_of::<E>(),
            "Slices must have same number of elements as *physical* Storage<E> of tensors."
        );
        src.data
            .data
            .copy_to_host(&src.device.dev, &src.device.queue, dst);
    }
}

impl<E: Unit> TensorFromVec<E> for Webgpu {
    fn try_tensor_from_vec<S: Shape>(
        &self,
        src: Vec<E>,
        shape: S,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let num_elements = shape.num_elements();

        if src.len() != num_elements {
            Err(Error::WrongNumElements)
        } else {
            self.tensor_from_host_buf(shape, src)
        }
    }
}

impl<S: Shape, E: Unit> TensorToArray<S, E> for Webgpu
where
    Cpu: TensorToArray<S, E> + Storage<E>,
{
    type Array = <Cpu as TensorToArray<S, E>>::Array;
    fn tensor_to_array<T>(&self, tensor: &Tensor<S, E, Self, T>) -> Self::Array {
        let buf = tensor.as_vec();
        let cpu_tensor = self.cpu.tensor_from_vec(buf, tensor.shape);
        self.cpu.tensor_to_array::<NoneTape>(&cpu_tensor)
    }
}
