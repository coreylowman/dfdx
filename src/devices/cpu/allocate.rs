#![allow(clippy::needless_range_loop)]

use crate::arrays::{
    Dtype, Rank0, Rank1, Rank2, Rank3, Rank4, Shape, StridesFor, TryFromNumElements,
};
use crate::devices::{
    AsArray, AsVec, Ones, OnesLike, Rand, RandLike, Randn, RandnLike, TryConvert, Zeros, ZerosLike,
};
use rand::Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use std::sync::Arc;
use std::vec::Vec;

use super::{
    device::{CpuError, StridedArray},
    iterate::LendingIterator,
    Cpu,
};

impl<S: Shape, E: Dtype> StridedArray<S, E> {
    #[inline]
    pub(super) fn try_new_with(shape: S, elem: E) -> Result<Self, CpuError> {
        let numel = shape.num_elements();
        let strides: StridesFor<S> = shape.strides();
        let mut data: Vec<E> = Vec::new();
        data.try_reserve_exact(numel)
            .map_err(|_| CpuError::OutOfMemory)?;
        data.resize(numel, elem);
        let data = Arc::new(data);
        Ok(StridedArray {
            data,
            shape,
            strides,
        })
    }

    #[inline]
    pub(super) fn try_new_like(other: &Self, elem: E) -> Result<Self, CpuError> {
        let numel = other.data.len();
        let shape = other.shape;
        let strides = other.strides;
        let mut data: Vec<E> = Vec::new();
        data.try_reserve_exact(numel)
            .map_err(|_| CpuError::OutOfMemory)?;
        data.resize(numel, elem);
        let data = Arc::new(data);
        Ok(StridedArray {
            data,
            shape,
            strides,
        })
    }

    #[inline]
    fn try_filled_with<R, D>(shape: S, rng: &mut R, dist: &D) -> Result<Self, CpuError>
    where
        R: Rng,
        D: Distribution<E>,
    {
        let numel = shape.num_elements();
        let strides: StridesFor<S> = shape.strides();
        let mut data: Vec<E> = Vec::new();
        data.try_reserve_exact(numel)
            .map_err(|_| CpuError::OutOfMemory)?;
        data.resize_with(numel, &mut || rng.sample(dist));
        Ok(StridedArray {
            data: Arc::new(data),
            shape,
            strides,
        })
    }

    #[inline]
    fn try_filled_like<R, D>(other: &Self, rng: &mut R, dist: &D) -> Result<Self, CpuError>
    where
        R: Rng,
        D: Distribution<E>,
    {
        let numel = other.data.len();
        let shape = other.shape;
        let strides = other.strides;
        let mut data: Vec<E> = Vec::new();
        data.try_reserve_exact(numel)
            .map_err(|_| CpuError::OutOfMemory)?;
        data.resize_with(numel, &mut || rng.sample(dist));
        Ok(StridedArray {
            data: Arc::new(data),
            shape,
            strides,
        })
    }
}

impl<S: Shape + Default, E: Dtype> Zeros<StridedArray<S, E>> for Cpu {
    #[inline]
    fn try_zeros(&self) -> Result<StridedArray<S, E>, Self::Err> {
        StridedArray::try_new_with(Default::default(), Default::default())
    }
    fn fill_with_zeros(&self, t: &mut StridedArray<S, E>) {
        let data = Arc::make_mut(&mut t.data);
        data.fill(Default::default());
    }
}

impl<S: Shape, E: Dtype> ZerosLike<&StridedArray<S, E>, StridedArray<S, E>> for Cpu {
    fn try_zeros_like(&self, src: &StridedArray<S, E>) -> Result<StridedArray<S, E>, Self::Err> {
        StridedArray::try_new_like(src, Default::default())
    }
}

impl<S: Shape, E: Dtype> ZerosLike<S, StridedArray<S, E>> for Cpu {
    fn try_zeros_like(&self, src: S) -> Result<StridedArray<S, E>, Self::Err> {
        StridedArray::try_new_with(src, Default::default())
    }
}

impl<S: Shape + Default> Ones<StridedArray<S, f32>> for Cpu {
    fn try_ones(&self) -> Result<StridedArray<S, f32>, Self::Err> {
        StridedArray::try_new_with(Default::default(), 1.0)
    }
    fn fill_with_ones(&self, t: &mut StridedArray<S, f32>) {
        let data = Arc::make_mut(&mut t.data);
        data.fill(1.0);
    }
}

impl<S: Shape> OnesLike<&StridedArray<S, f32>, StridedArray<S, f32>> for Cpu {
    fn try_ones_like(&self, src: &StridedArray<S, f32>) -> Result<StridedArray<S, f32>, Self::Err> {
        StridedArray::try_new_like(src, 1.0)
    }
}

impl<S: Shape> OnesLike<S, StridedArray<S, f32>> for Cpu {
    fn try_ones_like(&self, src: S) -> Result<StridedArray<S, f32>, Self::Err> {
        StridedArray::try_new_with(src, 1.0)
    }
}

impl<S: Shape + Default, E: Dtype> Rand<StridedArray<S, E>> for Cpu
where
    Standard: Distribution<E>,
{
    fn try_rand(&self) -> Result<StridedArray<S, E>, Self::Err> {
        let rng = &mut *self.rng.borrow_mut();
        StridedArray::try_filled_with(Default::default(), rng, &Standard)
    }
    fn fill_with_rand(&self, t: &mut StridedArray<S, E>) {
        let rng = &mut *self.rng.borrow_mut();
        let data = Arc::make_mut(&mut t.data);
        data.fill_with(&mut || rng.sample(Standard));
    }
}

impl<S: Shape, E: Dtype> RandLike<&StridedArray<S, E>, StridedArray<S, E>> for Cpu
where
    Standard: Distribution<E>,
{
    fn try_rand_like(&self, src: &StridedArray<S, E>) -> Result<StridedArray<S, E>, Self::Err> {
        let rng = &mut *self.rng.borrow_mut();
        StridedArray::try_filled_like(src, rng, &Standard)
    }
}

impl<S: Shape, E: Dtype> RandLike<S, StridedArray<S, E>> for Cpu
where
    Standard: Distribution<E>,
{
    fn try_rand_like(&self, src: S) -> Result<StridedArray<S, E>, Self::Err> {
        let rng = &mut *self.rng.borrow_mut();
        StridedArray::try_filled_with(src, rng, &Standard)
    }
}

impl<S: Shape + Default, E: Dtype> Randn<StridedArray<S, E>> for Cpu
where
    StandardNormal: Distribution<E>,
{
    fn try_randn(&self) -> Result<StridedArray<S, E>, Self::Err> {
        let rng = &mut *self.rng.borrow_mut();
        StridedArray::try_filled_with(Default::default(), rng, &StandardNormal)
    }
    fn fill_with_randn(&self, t: &mut StridedArray<S, E>) {
        let rng = &mut *self.rng.borrow_mut();
        let data = Arc::make_mut(&mut t.data);
        data.fill_with(&mut || rng.sample(StandardNormal));
    }
}

impl<S: Shape, E: Dtype> RandnLike<&StridedArray<S, E>, StridedArray<S, E>> for Cpu
where
    StandardNormal: Distribution<E>,
{
    fn try_randn_like(&self, src: &StridedArray<S, E>) -> Result<StridedArray<S, E>, Self::Err> {
        let rng = &mut *self.rng.borrow_mut();
        StridedArray::try_filled_like(src, rng, &StandardNormal)
    }
}

impl<S: Shape, E: Dtype> RandnLike<S, StridedArray<S, E>> for Cpu
where
    StandardNormal: Distribution<E>,
{
    fn try_randn_like(&self, src: S) -> Result<StridedArray<S, E>, Self::Err> {
        let rng = &mut *self.rng.borrow_mut();
        StridedArray::try_filled_with(src, rng, &StandardNormal)
    }
}

impl<S: Shape + TryFromNumElements, E: Dtype> TryConvert<&[E], StridedArray<S, E>> for Cpu {
    fn try_convert(&self, src: &[E]) -> Result<StridedArray<S, E>, Self::Err> {
        match S::try_from_num_elements(src.len()) {
            Some(shape) => {
                let mut storage: StridedArray<S, E> =
                    StridedArray::try_new_with(shape, Default::default())?;
                let data = Arc::make_mut(&mut storage.data);
                data.copy_from_slice(src);
                Ok(storage)
            }
            None => Err(CpuError::ShapeMismatch),
        }
    }
}

impl<S: Shape + TryFromNumElements, E: Dtype> TryConvert<Vec<E>, StridedArray<S, E>> for Cpu {
    fn try_convert(&self, src: Vec<E>) -> Result<StridedArray<S, E>, Self::Err> {
        match S::try_from_num_elements(src.len()) {
            Some(shape) => Ok(StridedArray {
                data: Arc::new(src),
                shape,
                strides: shape.strides(),
            }),
            None => Err(CpuError::ShapeMismatch),
        }
    }
}

impl<E: Dtype> TryConvert<E, StridedArray<Rank0, E>> for Cpu {
    fn try_convert(&self, src: E) -> Result<StridedArray<Rank0, E>, Self::Err> {
        let mut out: StridedArray<Rank0, E> = self.try_zeros()?;
        out[[]].clone_from(&src);
        Ok(out)
    }
}

impl<E: Dtype, const M: usize> TryConvert<[E; M], StridedArray<Rank1<M>, E>> for Cpu {
    fn try_convert(&self, src: [E; M]) -> Result<StridedArray<Rank1<M>, E>, Self::Err> {
        let mut out: StridedArray<Rank1<M>, E> = self.try_zeros()?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((v, [m])) = out_iter.next() {
            v.clone_from(&src[m]);
        }
        Ok(out)
    }
}

impl<E: Dtype, const M: usize, const N: usize> TryConvert<[[E; N]; M], StridedArray<Rank2<M, N>, E>>
    for Cpu
{
    fn try_convert(&self, src: [[E; N]; M]) -> Result<StridedArray<Rank2<M, N>, E>, Self::Err> {
        let mut out: StridedArray<Rank2<M, N>, E> = self.try_zeros()?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((v, [m, n])) = out_iter.next() {
            v.clone_from(&src[m][n]);
        }
        Ok(out)
    }
}

impl<E: Dtype, const M: usize, const N: usize, const O: usize>
    TryConvert<[[[E; O]; N]; M], StridedArray<Rank3<M, N, O>, E>> for Cpu
{
    fn try_convert(
        &self,
        src: [[[E; O]; N]; M],
    ) -> Result<StridedArray<Rank3<M, N, O>, E>, Self::Err> {
        let mut out: StridedArray<Rank3<M, N, O>, E> = self.try_zeros()?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((v, [m, n, o])) = out_iter.next() {
            v.clone_from(&src[m][n][o]);
        }
        Ok(out)
    }
}

impl<E: Dtype, const M: usize, const N: usize, const O: usize, const P: usize>
    TryConvert<[[[[E; P]; O]; N]; M], StridedArray<Rank4<M, N, O, P>, E>> for Cpu
{
    fn try_convert(
        &self,
        src: [[[[E; P]; O]; N]; M],
    ) -> Result<StridedArray<Rank4<M, N, O, P>, E>, Self::Err> {
        let mut out: StridedArray<Rank4<M, N, O, P>, E> = self.try_zeros()?;
        let mut out_iter = out.iter_mut_with_index();
        while let Some((v, [m, n, o, p])) = out_iter.next() {
            v.clone_from(&src[m][n][o][p]);
        }
        Ok(out)
    }
}

impl<const N: usize, S: Shape<Concrete = [usize; N]>, E: Dtype> AsVec for StridedArray<S, E> {
    type Vec = Vec<E>;
    fn as_vec(&self) -> Self::Vec {
        let numel = self.shape.num_elements();
        let mut out = Vec::with_capacity(numel);
        let mut iter = self.iter();
        while let Some(x) = iter.next() {
            out.push(*x);
        }
        out
    }
}

impl<E: Dtype> AsArray for StridedArray<Rank0, E> {
    type Array = E;
    fn as_array(&self) -> Self::Array {
        let mut out: Self::Array = Default::default();
        out.clone_from(&self.data[0]);
        out
    }
}

impl<E: Dtype, const M: usize> AsArray for StridedArray<Rank1<M>, E>
where
    [E; M]: Default,
{
    type Array = [E; M];
    fn as_array(&self) -> Self::Array {
        let mut out: Self::Array = Default::default();
        let mut iter = self.iter();
        for m in 0..M {
            out[m].clone_from(iter.next().unwrap());
        }
        out
    }
}

impl<E: Dtype, const M: usize, const N: usize> AsArray for StridedArray<Rank2<M, N>, E>
where
    [[E; N]; M]: Default,
{
    type Array = [[E; N]; M];
    fn as_array(&self) -> Self::Array {
        let mut out: Self::Array = Default::default();
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
where
    [[[E; O]; N]; M]: Default,
{
    type Array = [[[E; O]; N]; M];
    fn as_array(&self) -> Self::Array {
        let mut out: Self::Array = Default::default();
        let mut iter = self.iter_with_index();
        while let Some((v, [m, n, o])) = iter.next() {
            out[m][n][o].clone_from(v);
        }
        out
    }
}

impl<E: Dtype, const M: usize, const N: usize, const O: usize, const P: usize> AsArray
    for StridedArray<Rank4<M, N, O, P>, E>
where
    [[[[E; P]; O]; N]; M]: Default,
{
    type Array = [[[[E; P]; O]; N]; M];
    fn as_array(&self) -> Self::Array {
        let mut out: Self::Array = Default::default();
        let mut iter = self.iter_with_index();
        while let Some((v, [m, n, o, p])) = iter.next() {
            out[m][n][o][p].clone_from(v);
        }
        out
    }
}
