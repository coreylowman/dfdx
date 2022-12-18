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
use rand_distr::{Normal, Uniform};
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

impl OneFillStorage<f32> for Cuda {
    fn try_fill_with_ones<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        self.dev.copy_into_async(
            std::vec![1.0; storage.data.len()],
            Arc::make_mut(&mut storage.data),
        )?;
        Ok(())
    }
}

impl<E: Unit> RandTensor<E> for Cuda
where
    Cpu: RandTensor<E>,
{
    fn try_rand_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        self.take_cpu_tensor(self.cpu.try_rand_like(src)?)
    }
    fn try_uniform_like<S: HasShape>(
        &self,
        src: &S,
        min: E,
        max: E,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        self.take_cpu_tensor(self.cpu.try_uniform_like(src, min, max)?)
    }
}

impl RandFillStorage<f32> for Cuda {
    fn try_fill_with_uniform<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, f32>,
        min: f32,
        max: f32,
    ) -> Result<(), Self::Err> {
        let dist = Uniform::new(min, max);
        let mut host_data = std::vec![0.0; storage.data.len()];
        {
            let mut rng = self.cpu.rng.lock().unwrap();
            host_data.fill_with(|| rng.sample(dist));
        }
        self.dev
            .copy_into_async(host_data, Arc::make_mut(&mut storage.data))?;
        Ok(())
    }
}

impl<E: Unit> RandnTensor<E> for Cuda
where
    Cpu: RandnTensor<E>,
{
    fn try_randn_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        self.take_cpu_tensor(self.cpu.try_randn_like(src)?)
    }
    fn try_normal_like<S: HasShape>(
        &self,
        src: &S,
        mean: E,
        stddev: E,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        self.take_cpu_tensor(self.cpu.try_normal_like(src, mean, stddev)?)
    }
}

impl RandnFillStorage<f32> for Cuda {
    fn try_fill_with_normal<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, f32>,
        mean: f32,
        stddev: f32,
    ) -> Result<(), Self::Err> {
        let dist = Normal::new(mean, stddev).unwrap();
        let mut host_vec = std::vec![0.0; storage.data.len()];
        {
            let mut rng = self.cpu.rng.lock().unwrap();
            host_vec.fill_with(|| rng.sample(dist));
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
