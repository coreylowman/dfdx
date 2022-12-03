#![allow(clippy::needless_range_loop)]

use crate::arrays::*;
use crate::tensor::storage::*;
use crate::tensor::{DeviceStorage, Tensor, TensorFromStorage};
use rand::Rng;
use rand_distr::{Normal, Standard, StandardNormal, Uniform};
use std::{sync::Arc, vec::Vec};

use super::{
    device::{CpuError, StridedArray},
    iterate::LendingIterator,
    Cpu,
};

impl<S: Shape, E: Dtype> StridedArray<S, E> {
    #[inline]
    pub(crate) fn new(shape: S) -> Result<Self, CpuError> {
        Self::try_new_with(shape, Default::default())
    }

    #[inline]
    pub(crate) fn try_new_with(shape: S, elem: E) -> Result<Self, CpuError> {
        let numel = shape.num_elements();
        let strides: S::Concrete = shape.strides();
        let mut data: Vec<E> = Vec::new();
        data.try_reserve(numel).map_err(|_| CpuError::OutOfMemory)?;
        data.resize(numel, elem);
        let data = Arc::new(data);
        Ok(StridedArray {
            data,
            shape,
            strides,
        })
    }

    #[inline]
    pub(crate) fn try_new_like(other: &Self, elem: E) -> Result<Self, CpuError> {
        let numel = other.data.len();
        let shape = other.shape;
        let strides = other.strides;
        let mut data: Vec<E> = Vec::new();
        data.try_reserve(numel).map_err(|_| CpuError::OutOfMemory)?;
        data.resize(numel, elem);
        let data = Arc::new(data);
        Ok(StridedArray {
            data,
            shape,
            strides,
        })
    }
}

impl<E: Dtype> ZerosTensor<E> for Cpu {
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let storage = StridedArray::try_new_with(*src.shape(), Default::default())?;
        Ok(self.upgrade(storage))
    }
}

impl<E: Dtype> ZeroFillStorage<E> for Cpu {
    fn try_fill_with_zeros<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        std::sync::Arc::make_mut(&mut storage.data).fill(Default::default());
        Ok(())
    }
}

impl OnesTensor<f32> for Cpu {
    fn try_ones_like<S: HasShape>(
        &self,
        src: &S,
    ) -> Result<Tensor<S::Shape, f32, Self>, Self::Err> {
        let storage = StridedArray::try_new_with(*src.shape(), 1.0)?;
        Ok(self.upgrade(storage))
    }
}

impl OneFillStorage<f32> for Cpu {
    fn try_fill_with_ones<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        std::sync::Arc::make_mut(&mut storage.data).fill(1.0);
        Ok(())
    }
}

impl RandTensor<f32> for Cpu {
    fn try_rand_like<S: HasShape>(
        &self,
        src: &S,
    ) -> Result<Tensor<S::Shape, f32, Self>, Self::Err> {
        let mut storage = StridedArray::try_new_with(*src.shape(), Default::default())?;
        {
            let mut rng = self.rng.lock().unwrap();
            for v in storage.buf_iter_mut() {
                *v = rng.sample(Standard);
            }
        }
        Ok(self.upgrade(storage))
    }
    fn try_uniform_like<S: HasShape>(
        &self,
        src: &S,
        min: f32,
        max: f32,
    ) -> Result<Tensor<S::Shape, f32, Self>, Self::Err> {
        let mut storage = StridedArray::try_new_with(*src.shape(), Default::default())?;
        let dist = Uniform::new(min, max);
        {
            let mut rng = self.rng.lock().unwrap();
            for v in storage.buf_iter_mut() {
                *v = rng.sample(dist);
            }
        }
        Ok(self.upgrade(storage))
    }
}

impl RandFillStorage<f32> for Cpu {
    fn try_fill_with_uniform<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, f32>,
        min: f32,
        max: f32,
    ) -> Result<(), Self::Err> {
        let dist = Uniform::new(min, max);
        {
            let mut rng = self.rng.lock().unwrap();
            for v in storage.buf_iter_mut() {
                *v = rng.sample(dist);
            }
        }
        Ok(())
    }
}

impl RandnTensor<f32> for Cpu {
    fn try_randn_like<S: HasShape>(
        &self,
        src: &S,
    ) -> Result<Tensor<S::Shape, f32, Self>, Self::Err> {
        let mut storage = StridedArray::try_new_with(*src.shape(), Default::default())?;
        {
            let mut rng = self.rng.lock().unwrap();
            for v in storage.buf_iter_mut() {
                *v = rng.sample(StandardNormal);
            }
        }
        Ok(self.upgrade(storage))
    }
    fn try_normal_like<S: HasShape>(
        &self,
        src: &S,
        mean: f32,
        stddev: f32,
    ) -> Result<Tensor<S::Shape, f32, Self>, Self::Err> {
        let mut storage = StridedArray::try_new_with(*src.shape(), Default::default())?;
        let dist = Normal::new(mean, stddev).unwrap();
        {
            let mut rng = self.rng.lock().unwrap();
            for v in storage.buf_iter_mut() {
                *v = rng.sample(dist);
            }
        }
        Ok(self.upgrade(storage))
    }
}

impl RandnFillStorage<f32> for Cpu {
    fn try_fill_with_normal<S: Shape>(
        &self,
        storage: &mut Self::Storage<S, f32>,
        mean: f32,
        stddev: f32,
    ) -> Result<(), Self::Err> {
        let dist = Normal::new(mean, stddev).unwrap();
        {
            let mut rng = self.rng.lock().unwrap();
            for v in storage.buf_iter_mut() {
                *v = rng.sample(dist);
            }
        }
        Ok(())
    }
}

impl<E: Dtype> TensorFromSlice<E> for Cpu {
    fn try_from_slice<S: Shape + TryFromNumElements>(
        &self,
        src: &[E],
    ) -> Option<Result<Tensor<S, E, Self>, Self::Err>> {
        S::try_from_num_elements(src.len()).map(|shape| {
            let mut storage = StridedArray::try_new_with(shape, Default::default())?;
            std::sync::Arc::make_mut(&mut storage.data).copy_from_slice(src);
            Ok(self.upgrade(storage))
        })
    }
}

impl<E: Dtype> TensorFromVec<E> for Cpu {
    fn try_from_vec<S: Shape + TryFromNumElements>(
        &self,
        src: std::vec::Vec<E>,
    ) -> Option<Result<Tensor<S, E, Self>, Self::Err>> {
        S::try_from_num_elements(src.len()).map(|shape| {
            let storage = StridedArray {
                data: std::sync::Arc::new(src),
                shape,
                strides: shape.strides(),
            };
            Ok(self.upgrade(storage))
        })
    }
}

impl<E: Dtype> TensorFromArray<E, Rank0, E> for Cpu {
    fn try_from_array(&self, src: E) -> Result<Tensor<Rank0, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank0, E> = self.try_alloc(&Default::default())?;
        storage[[]].clone_from(&src);
        Ok(self.upgrade(storage))
    }
}

impl<E: Dtype, const M: usize> TensorFromArray<[E; M], Rank1<M>, E> for Cpu {
    fn try_from_array(&self, src: [E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank1<M>, E> = self.try_alloc(&Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m])) = iter.next() {
            v.clone_from(&src[m]);
        }
        Ok(self.upgrade(storage))
    }
}

impl<E: Dtype, const M: usize, const N: usize> TensorFromArray<[[E; N]; M], Rank2<M, N>, E>
    for Cpu
{
    fn try_from_array(&self, src: [[E; N]; M]) -> Result<Tensor<Rank2<M, N>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank2<M, N>, E> = self.try_alloc(&Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m, n])) = iter.next() {
            v.clone_from(&src[m][n]);
        }
        Ok(self.upgrade(storage))
    }
}

impl<E: Dtype, const M: usize, const N: usize, const O: usize>
    TensorFromArray<[[[E; O]; N]; M], Rank3<M, N, O>, E> for Cpu
{
    fn try_from_array(
        &self,
        src: [[[E; O]; N]; M],
    ) -> Result<Tensor<Rank3<M, N, O>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank3<M, N, O>, E> = self.try_alloc(&Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m, n, o])) = iter.next() {
            v.clone_from(&src[m][n][o]);
        }
        Ok(self.upgrade(storage))
    }
}

impl<E: Dtype, const M: usize, const N: usize, const O: usize, const P: usize>
    TensorFromArray<[[[[E; P]; O]; N]; M], Rank4<M, N, O, P>, E> for Cpu
{
    fn try_from_array(
        &self,
        src: [[[[E; P]; O]; N]; M],
    ) -> Result<Tensor<Rank4<M, N, O, P>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank4<M, N, O, P>, E> = self.try_alloc(&Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m, n, o, p])) = iter.next() {
            v.clone_from(&src[m][n][o][p]);
        }
        Ok(self.upgrade(storage))
    }
}

impl<S: Shape, E: Dtype> AsVec for StridedArray<S, E> {
    type Vec = Vec<E>;
    fn as_vec(&self) -> Self::Vec {
        let mut out = Vec::with_capacity(self.shape.num_elements());
        let mut iter = self.iter();
        while let Some(x) = iter.next() {
            out.push(*x);
        }
        out
    }
}

impl<E: Dtype> AsArray for StridedArray<Rank0, E> {
    type Array = E;
    fn array(&self) -> Self::Array {
        let mut out: Self::Array = Default::default();
        out.clone_from(&self.data[0]);
        out
    }
}

impl<E: Dtype, const M: usize> AsArray for StridedArray<Rank1<M>, E> {
    type Array = [E; M];
    fn array(&self) -> Self::Array {
        let mut out: Self::Array = [Default::default(); M];
        let mut iter = self.iter();
        for m in 0..M {
            out[m].clone_from(iter.next().unwrap());
        }
        out
    }
}

impl<E: Dtype, const M: usize, const N: usize> AsArray for StridedArray<Rank2<M, N>, E> {
    type Array = [[E; N]; M];
    fn array(&self) -> Self::Array {
        let mut out: Self::Array = [[Default::default(); N]; M];
        let mut iter = self.iter();
        for m in 0..M {
            for n in 0..N {
                out[m][n].clone_from(iter.next().unwrap());
            }
        }
        out
    }
}

impl<E: Dtype, const M: usize, const N: usize, const O: usize> AsArray
    for StridedArray<Rank3<M, N, O>, E>
{
    type Array = [[[E; O]; N]; M];
    fn array(&self) -> Self::Array {
        let mut out: Self::Array = [[[Default::default(); O]; N]; M];
        let mut iter = self.iter_with_index();
        while let Some((v, [m, n, o])) = iter.next() {
            out[m][n][o].clone_from(v);
        }
        out
    }
}

impl<E: Dtype, const M: usize, const N: usize, const O: usize, const P: usize> AsArray
    for StridedArray<Rank4<M, N, O, P>, E>
{
    type Array = [[[[E; P]; O]; N]; M];
    fn array(&self) -> Self::Array {
        let mut out: Self::Array = [[[[Default::default(); P]; O]; N]; M];
        let mut iter = self.iter_with_index();
        while let Some((v, [m, n, o, p])) = iter.next() {
            out[m][n][o][p].clone_from(v);
        }
        out
    }
}
