#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::*,
    tensor::{
        cpu::{Cpu, StridedArray},
        storage_traits::*,
        Tensor,
    },
};

use super::{Cuda, CudaArray, CudaError};

use rand::Rng;
use std::{sync::Arc, vec::Vec};

use num_traits::One;

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

impl<E: Unit> ZerosTensor<E> for Cuda
where
    Cpu: ZerosTensor<E>,
{
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        self.take_cpu_tensor(self.cpu.try_zeros_like(src)?)
    }
}

impl<E: Unit> ZeroFillStorage<E> for Cuda {
    fn try_fill_with_zeros<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        self.dev.copy_into_async(
            std::vec![Default::default(); storage.data.len()],
            Arc::make_mut(&mut storage.data),
        )?;
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

impl<E: Unit + One> OneFillStorage<E> for Cuda {
    fn try_fill_with_ones<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        self.dev.copy_into_async(
            std::vec![One::one(); storage.data.len()],
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
        dst.device
            .dev
            .sync_copy_into(src, Arc::make_mut(&mut dst.storage.data))
            .unwrap();
    }
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]) {
        src.device
            .dev
            .sync_copy_from(src.storage.data.as_ref(), dst)
            .unwrap();
    }
}

impl<S: Shape, E: Unit> AsVec for CudaArray<S, E> {
    fn as_vec(&self) -> Vec<E> {
        self.data.clone_async().unwrap().try_into().unwrap()
    }
}

impl<Src, S: Shape, E: Unit> TensorFromArray<Src, S, E> for Cuda
where
    Cpu: TensorFromArray<Src, S, E>,
{
    fn try_tensor(&self, src: Src) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.take_cpu_tensor(self.cpu.try_tensor(src)?)
    }
}

impl<S: Shape, E: Unit> AsArray for CudaArray<S, E>
where
    StridedArray<S, E>: AsArray,
{
    type Array = <StridedArray<S, E> as AsArray>::Array;
    fn array(&self) -> Self::Array {
        let a = StridedArray {
            data: Arc::new(self.as_vec()),
            shape: self.shape,
            strides: self.strides,
        };
        a.array()
    }
}
