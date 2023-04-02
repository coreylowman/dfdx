#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::*,
    tensor::{masks::triangle_mask, storage_traits::*, unique_id, Tensor},
};

use super::{Cpu, CpuError, LendingIterator};

use rand::{distributions::Distribution, Rng};
use std::{sync::Arc, vec::Vec};

impl<G> Cpu<G> {
    #[inline]
    pub(crate) fn try_alloc_zeros<E: Unit>(&self, numel: usize) -> Result<G::Storage, CpuError>
    where
        G: DeviceStorage<E>,
    {
        Ok(G::Storage::try_alloc_zeros(numel).map_err(|_| CpuError::OutOfMemory)?)
    }

    #[inline]
    pub(crate) fn try_alloc_elem<E: Unit>(
        &self,
        numel: usize,
        elem: E,
    ) -> Result<G::Storage, CpuError>
    where
        G: DeviceStorage<E>,
    {
        Ok(G::Storage::try_alloc_elem(numel, elem).map_err(|_| CpuError::OutOfMemory)?)
    }
}

impl<E: Unit, G: DeviceStorage<E> + Clone> ZerosTensor<E> for Cpu<G> {
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

impl<E: Unit, G: DeviceStorage<E>> ZeroFillStorage<E> for Cpu<G> {
    fn try_fill_with_zeros(&self, storage: &mut G::Storage) -> Result<(), Self::Err> {
        storage.fill(Default::default());
        Ok(())
    }
}

impl<E: Unit, G: DeviceStorage<E> + Clone> OnesTensor<E> for Cpu<G> {
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

impl<E: Unit, G: DeviceStorage<E>> OneFillStorage<E> for Cpu<G> {
    fn try_fill_with_ones(&self, storage: &mut G::Storage) -> Result<(), Self::Err> {
        storage.fill(E::ONE);
        Ok(())
    }
}

impl<E: Unit, G: DeviceStorage<E> + Clone> SampleTensor<E> for Cpu<G> {
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
            let mut iter = Arc::get_mut(&mut tensor.data).unwrap().iter_mut();
            // while let Some(v) = iter.next() {
            //     *v = rng.sample(&distr);
            // }
            todo!()
        }
        Ok(tensor)
    }
    fn try_fill_with_distr<D: Distribution<E>>(
        &self,
        storage: &mut G::Storage,
        distr: D,
    ) -> Result<(), Self::Err> {
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.rng.lock();
            let mut iter = storage.iter_mut();
            // while let Some(v) = iter.next() {
            //     *v = rng.sample(&distr);
            // }
            todo!()
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

impl<E: Unit, G: DeviceStorage<E> + Clone> TensorFromVec<E> for Cpu<G> {
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
                data: Arc::new(G::Storage::from_vec(src)),
                shape,
                strides: shape.strides(),
                device: self.clone(),
                tape: Default::default(),
            })
        }
    }
}

impl<E: Unit, G: DeviceStorage<E>> TensorToArray<E, Rank0> for Cpu<G> {
    type Array = E;
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank0, E, Self, T>) -> Self::Array {
        tensor.data.index(0)
    }
}

impl<E: Unit, G: DeviceStorage<E>, const M: usize> TensorToArray<E, Rank1<M>> for Cpu<G> {
    type Array = [E; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank1<M>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [Default::default(); M];
        for (v, [m]) in tensor.iter_copied_with_index() {
            out[m] = v;
        }
        out
    }
}

impl<E: Unit, G: DeviceStorage<E>, const M: usize, const N: usize> TensorToArray<E, Rank2<M, N>>
    for Cpu<G>
{
    type Array = [[E; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank2<M, N>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[Default::default(); N]; M];
        for (v, [m, n]) in tensor.iter_copied_with_index() {
            out[m][n] = v;
        }
        out
    }
}

impl<E: Unit, G: DeviceStorage<E>, const M: usize, const N: usize, const O: usize>
    TensorToArray<E, Rank3<M, N, O>> for Cpu<G>
{
    type Array = [[[E; O]; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank3<M, N, O>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[[Default::default(); O]; N]; M];
        for (v, [m, n, o]) in tensor.iter_copied_with_index() {
            out[m][n][o] = v;
        }
        out
    }
}

impl<
        E: Unit,
        G: DeviceStorage<E>,
        const M: usize,
        const N: usize,
        const O: usize,
        const P: usize,
    > TensorToArray<E, Rank4<M, N, O, P>> for Cpu<G>
{
    type Array = [[[[E; P]; O]; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank4<M, N, O, P>, E, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[[[Default::default(); P]; O]; N]; M];
        for (v, [m, n, o, p]) in tensor.iter_copied_with_index() {
            out[m][n][o][p] = v;
        }
        out
    }
}
