#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::*,
    tensor::{masks::triangle_mask, storage_traits::*, unique_id, Tensor},
};

use super::{Cpu, CpuError, LendingIterator};

use rand::{distributions::Distribution, Rng};
use std::{sync::Arc, vec::Vec};

impl Cpu {
    #[inline]
    pub(crate) fn try_alloc_zeros<E: Unit>(&self, numel: usize) -> Result<Vec<E>, CpuError> {
        self.try_alloc_elem::<E>(numel, Default::default())
    }

    #[inline]
    pub(crate) fn try_alloc_elem<E: Unit>(
        &self,
        numel: usize,
        elem: E,
    ) -> Result<Vec<E>, CpuError> {
        #[cfg(feature = "fast-alloc")]
        {
            Ok(std::vec![elem; numel])
        }

        #[cfg(not(feature = "fast-alloc"))]
        {
            let mut data: Vec<E> = Vec::new();
            data.try_reserve(numel).map_err(|_| CpuError::OutOfMemory)?;
            data.resize(numel, elem);
            Ok(data)
        }
    }
}

impl<E: Unit> ZerosTensor<E> for Cpu {
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let data = self.try_alloc_zeros::<E>(shape.num_elements())?;
        let data = Arc::new(data);
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
}

impl<E: Unit> ZeroFillStorage<E> for Cpu {
    fn try_fill_with_zeros(&self, storage: &mut Self::Storage) -> Result<(), Self::Err> {
        storage.fill(Default::default());
        Ok(())
    }
}

impl<E: Unit> OnesTensor<E> for Cpu {
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let data = self.try_alloc_elem::<E>(shape.num_elements(), E::ONE)?;
        let data = Arc::new(data);
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
}

impl<E: Unit> TriangleTensor<E> for Cpu {
    fn try_upper_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let mut data = self.try_alloc_elem::<E>(shape.num_elements(), val)?;
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(&mut data, &shape, true, offset);
        let data = Arc::new(data);
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }

    fn try_lower_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let mut data = self.try_alloc_elem::<E>(shape.num_elements(), val)?;
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(&mut data, &shape, false, offset);
        let data = Arc::new(data);
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
}

impl<E: Unit> OneFillStorage<E> for Cpu {
    fn try_fill_with_ones(&self, storage: &mut Self::Storage) -> Result<(), Self::Err> {
        storage.fill(E::ONE);
        Ok(())
    }
}

impl<E: Unit> SampleTensor<E> for Cpu {
    fn try_sample_like<S: HasShape, D: Distribution<E>>(
        &self,
        src: &S,
        distr: D,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let mut tensor = self.try_zeros_like(src)?;
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.rng.lock();
            for v in Arc::get_mut(&mut tensor.data).unwrap().iter_mut() {
                *v = rng.sample(&distr);
            }
        }
        Ok(tensor)
    }
    fn try_fill_with_distr<D: Distribution<E>>(
        &self,
        storage: &mut Self::Storage,
        distr: D,
    ) -> Result<(), Self::Err> {
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.rng.lock();
            for v in storage.iter_mut() {
                *v = rng.sample(&distr);
            }
        }
        Ok(())
    }
}

impl<E: Unit> CopySlice<E> for Cpu {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]) {
        std::sync::Arc::make_mut(&mut dst.data).copy_from_slice(src);
    }
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]) {
        dst.copy_from_slice(src.data.as_ref());
    }
}

impl<E: Unit> TensorFromVec<E> for Cpu {
    fn try_tensor_from_vec<S: Shape>(
        &self,
        src: Vec<E>,
        shape: S,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let num_elements = shape.num_elements();

        if src.len() != num_elements {
            Err(CpuError::WrongNumElements)
        } else {
            Ok(Tensor {
                id: unique_id(),
                data: Arc::new(src),
                shape,
                strides: shape.strides(),
                device: self.clone(),
                tape: Default::default(),
            })
        }
    }
}

impl<E: Unit> TensorToArray<Rank0, E> for Cpu {
    type Array = E;
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank0, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = Default::default();
        out.clone_from(&tensor.data[0]);
        out
    }
}

impl<E: Unit, const M: usize> TensorToArray<Rank1<M>, E> for Cpu {
    type Array = [E; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank1<M>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [Default::default(); M];
        let mut iter = tensor.iter();
        for m in 0..M {
            out[m].clone_from(iter.next().unwrap());
        }
        out
    }
}

impl<E: Unit, const M: usize, const N: usize> TensorToArray<Rank2<M, N>, E> for Cpu {
    type Array = [[E; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank2<M, N>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[Default::default(); N]; M];
        let mut iter = tensor.iter();
        for m in 0..M {
            for n in 0..N {
                out[m][n].clone_from(iter.next().unwrap());
            }
        }
        out
    }
}

impl<E: Unit, const M: usize, const N: usize, const O: usize> TensorToArray<Rank3<M, N, O>, E>
    for Cpu
{
    type Array = [[[E; O]; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank3<M, N, O>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[[Default::default(); O]; N]; M];
        let mut iter = tensor.iter_with_index();
        while let Some((v, [m, n, o])) = iter.next() {
            out[m][n][o].clone_from(v);
        }
        out
    }
}

impl<E: Unit, const M: usize, const N: usize, const O: usize, const P: usize>
    TensorToArray<Rank4<M, N, O, P>, E> for Cpu
{
    type Array = [[[[E; P]; O]; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank4<M, N, O, P>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[[[Default::default(); P]; O]; N]; M];
        let mut iter = tensor.iter_with_index();
        while let Some((v, [m, n, o, p])) = iter.next() {
            out[m][n][o][p].clone_from(v);
        }
        out
    }
}
