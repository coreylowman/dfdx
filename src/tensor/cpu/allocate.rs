#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::*,
    tensor::{storage_traits::*, Tensor},
};
use rand::{distributions::Distribution, Rng};
use std::{sync::Arc, vec::Vec};

use super::{Cpu, CpuError, LendingIterator, StridedArray};

impl<S: Shape, E: Default + Clone> StridedArray<S, E> {
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

impl<E: Unit> ZerosTensor<E> for Cpu {
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let storage = StridedArray::try_new_with(*src.shape(), Default::default())?;
        Ok(self.upgrade(storage))
    }
}

impl<E: Unit> ZeroFillStorage<E> for Cpu {
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

impl<E: Unit> SampleTensor<E> for Cpu {
    fn try_sample_like<S: HasShape, D: Distribution<E>>(
        &self,
        src: &S,
        distr: D,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let mut storage = StridedArray::try_new_with(*src.shape(), Default::default())?;
        {
            let mut rng = self.rng.lock().unwrap();
            for v in storage.buf_iter_mut() {
                *v = rng.sample(&distr);
            }
        }
        Ok(self.upgrade(storage))
    }
    fn try_fill_with_distr<S: Shape, D: Distribution<E>>(
        &self,
        storage: &mut Self::Storage<S, E>,
        distr: D,
    ) -> Result<(), Self::Err> {
        {
            let mut rng = self.rng.lock().unwrap();
            for v in storage.buf_iter_mut() {
                *v = rng.sample(&distr);
            }
        }
        Ok(())
    }
}

impl<E: Unit> CopySlice<E> for Cpu {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]) {
        std::sync::Arc::make_mut(&mut dst.storage.data).copy_from_slice(src);
    }
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]) {
        dst.copy_from_slice(src.storage.data.as_ref());
    }
}

impl<E: Unit> TensorFromArray<E, Rank0, E> for Cpu {
    fn try_tensor(&self, src: E) -> Result<Tensor<Rank0, E, Self>, Self::Err> {
        let mut storage: StridedArray<_, E> = StridedArray::new(Default::default())?;
        storage[[]].clone_from(&src);
        Ok(self.upgrade(storage))
    }
}

impl<E: Unit, const M: usize> TensorFromArray<[E; M], Rank1<M>, E> for Cpu {
    fn try_tensor(&self, src: [E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank1<M>, E> = StridedArray::new(Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m])) = iter.next() {
            v.clone_from(&src[m]);
        }
        Ok(self.upgrade(storage))
    }
}

#[test]
fn test_tensor_from() {
    let device = Cpu::default();
    let _tensor = device.tensor_with_shape(std::vec![1.0, 2.0], (2,));
}

#[allow(unused)]
fn test_tensor_from2<D: crate::prelude::Device<f32>>(device: D) {
    let _tensor = device.tensor_with_shape(std::vec![1.0, 2.0], (2,));
}

impl<E: Unit, const M: usize> TensorFromArray<&[E; M], Rank1<M>, E> for Cpu {
    fn try_tensor(&self, src: &[E; M]) -> Result<Tensor<Rank1<M>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank1<M>, E> = StridedArray::new(Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m])) = iter.next() {
            v.clone_from(&src[m]);
        }
        Ok(self.upgrade(storage))
    }
}

impl<E: Unit, const M: usize, const N: usize> TensorFromArray<[[E; N]; M], Rank2<M, N>, E> for Cpu {
    fn try_tensor(&self, src: [[E; N]; M]) -> Result<Tensor<Rank2<M, N>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank2<M, N>, E> = StridedArray::new(Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m, n])) = iter.next() {
            v.clone_from(&src[m][n]);
        }
        Ok(self.upgrade(storage))
    }
}

impl<E: Unit, const M: usize, const N: usize, const O: usize>
    TensorFromArray<[[[E; O]; N]; M], Rank3<M, N, O>, E> for Cpu
{
    fn try_tensor(
        &self,
        src: [[[E; O]; N]; M],
    ) -> Result<Tensor<Rank3<M, N, O>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank3<M, N, O>, E> = StridedArray::new(Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m, n, o])) = iter.next() {
            v.clone_from(&src[m][n][o]);
        }
        Ok(self.upgrade(storage))
    }
}

impl<E: Unit, const M: usize, const N: usize, const O: usize, const P: usize>
    TensorFromArray<[[[[E; P]; O]; N]; M], Rank4<M, N, O, P>, E> for Cpu
{
    fn try_tensor(
        &self,
        src: [[[[E; P]; O]; N]; M],
    ) -> Result<Tensor<Rank4<M, N, O, P>, E, Self>, Self::Err> {
        let mut storage: StridedArray<Rank4<M, N, O, P>, E> =
            StridedArray::new(Default::default())?;
        let mut iter = storage.iter_mut_with_index();
        while let Some((v, [m, n, o, p])) = iter.next() {
            v.clone_from(&src[m][n][o][p]);
        }
        Ok(self.upgrade(storage))
    }
}

impl<E: Unit> TensorFromVec<E> for Cpu {
    fn try_tensor_with_shape<S: Shape>(
        &self,
        mut src: Vec<E>,
        shape: S,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let num_elements = shape.num_elements();

        if src.len() < num_elements {
            // TODO: This error makes no sense
            Err(CpuError::OutOfMemory)
        } else {
            src.truncate(num_elements);
            let array = StridedArray {
                data: Arc::new(src),
                shape,
                strides: shape.strides(),
            };

            Ok(self.upgrade(array))
        }
    }
}

impl<S: Shape, E: Unit> AsVec for StridedArray<S, E> {
    fn as_vec(&self) -> Vec<E> {
        let mut out = Vec::with_capacity(self.shape.num_elements());
        let mut iter = self.iter();
        while let Some(x) = iter.next() {
            out.push(*x);
        }
        out
    }
}

impl<E: Unit> AsArray for StridedArray<Rank0, E> {
    type Array = E;
    fn array(&self) -> Self::Array {
        let mut out: Self::Array = Default::default();
        out.clone_from(&self.data[0]);
        out
    }
}

impl<E: Unit, const M: usize> AsArray for StridedArray<Rank1<M>, E> {
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

impl<E: Unit, const M: usize, const N: usize> AsArray for StridedArray<Rank2<M, N>, E> {
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

impl<E: Unit, const M: usize, const N: usize, const O: usize> AsArray
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

impl<E: Unit, const M: usize, const N: usize, const O: usize, const P: usize> AsArray
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
