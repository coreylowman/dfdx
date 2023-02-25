#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::*,
    tensor::{
        cpu::{Cpu, CpuError, StridedArray},
        storage_traits::*,
        Tensor,
    },
};

use super::{Cuda, CudaArray, CudaError};

use rand::Rng;
use std::{sync::Arc, vec::Vec};

impl Cuda {
    #[inline(always)]
    pub(crate) fn take_cpu_tensor<S: Shape, E: Unit>(
        &self,
        t_cpu: Tensor<S, E, Cpu>,
    ) -> Result<Tensor<S, E, Self>, CudaError> {
        let data = self
            .dev
            .take_async(Arc::try_unwrap(t_cpu.storage.data).unwrap())?;
        let storage = CudaArray {
            data: Arc::new(data),
            shape: t_cpu.storage.shape,
            strides: t_cpu.storage.strides,
        };
        Ok(Tensor {
            id: t_cpu.id,
            storage,
            tape: Default::default(),
            device: self.clone(),
        })
    }
}

impl<E: Unit> ZerosTensor<E> for Cuda {
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let data = self.dev.alloc_zeros_async(shape.num_elements())?;
        let storage = CudaArray {
            data: Arc::new(data),
            shape,
            strides,
        };
        Ok(self.upgrade(storage))
    }
}

impl<E: Unit> ZeroFillStorage<E> for Cuda {
    fn try_fill_with_zeros<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        self.dev
            .memset_zeros_async(Arc::make_mut(&mut storage.data))?;
        Ok(())
    }
}

impl<E: Unit> OnesTensor<E> for Cuda
where
    Cpu: OnesTensor<E>,
{
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        self.take_cpu_tensor(self.cpu.try_ones_like(src)?)
    }
}

impl<E: Unit> OneFillStorage<E> for Cuda {
    fn try_fill_with_ones<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        self.dev.copy_into_async(
            std::vec![E::ONE; storage.data.len()],
            Arc::make_mut(&mut storage.data),
        )?;
        Ok(())
    }
}

impl<E: Unit> SampleTensor<E> for Cuda
where
    Cpu: SampleTensor<E>,
{
    fn try_sample_like<S: HasShape, D: rand_distr::Distribution<E>>(
        &self,
        src: &S,
        distr: D,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        self.take_cpu_tensor(self.cpu.try_sample_like(src, distr)?)
    }
    fn try_fill_with_distr<S: Shape, D: rand_distr::Distribution<E>>(
        &self,
        storage: &mut Self::Storage<S, E>,
        distr: D,
    ) -> Result<(), Self::Err> {
        let mut host_vec = std::vec![Default::default(); storage.data.len()];
        {
            let mut rng = self.cpu.rng.lock().unwrap();
            host_vec.fill_with(|| rng.sample(&distr));
        }
        self.dev
            .copy_into_async(host_vec, Arc::make_mut(&mut storage.data))?;
        Ok(())
    }
}

impl<E: Unit> CopySlice<E> for Cuda {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]) {
        assert_eq!(
            dst.storage.data.len(),
            src.len(),
            "Slices must have same number of elements as *physical* storage of tensors."
        );
        dst.device
            .dev
            .sync_copy_into(src, Arc::make_mut(&mut dst.storage.data))
            .unwrap();
    }
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]) {
        assert_eq!(
            src.storage.data.len(),
            dst.len(),
            "Slices must have same number of elements as *physical* storage of tensors."
        );
        src.device
            .dev
            .sync_copy_from(src.storage.data.as_ref(), dst)
            .unwrap();
    }
}

impl<S: Shape, E: Unit> AsVec<E> for CudaArray<S, E> {
    fn as_vec(&self) -> Vec<E> {
        let buf = self.data.clone_async().unwrap().try_into().unwrap();
        let a = StridedArray {
            data: Arc::new(buf),
            shape: self.shape,
            strides: self.strides,
        };
        a.as_vec()
    }
}

impl<E: Unit> TensorFromVec<E> for Cuda {
    fn try_tensor_from_vec<S: Shape>(
        &self,
        src: Vec<E>,
        shape: S,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let num_elements = shape.num_elements();

        if src.len() != num_elements {
            Err(CudaError::Cpu(CpuError::WrongNumElements))
        } else {
            let array = CudaArray {
                data: Arc::new(self.dev.take_async(src)?),
                shape,
                strides: shape.strides(),
            };

            Ok(self.upgrade(array))
        }
    }
}

impl<S: Shape, E: Unit> AsArray for CudaArray<S, E>
where
    StridedArray<S, E>: AsArray,
{
    type Array = <StridedArray<S, E> as AsArray>::Array;
    fn array(&self) -> Self::Array {
        let buf = self.data.clone_async().unwrap().try_into().unwrap();
        let a = StridedArray {
            data: Arc::new(buf),
            shape: self.shape,
            strides: self.strides,
        };
        a.array()
    }
}
