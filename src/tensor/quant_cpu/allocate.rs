#![allow(clippy::needless_range_loop)]

use crate::{
    prelude::{cpu::LendingIterator, CpuError},
    shapes::*,
    tensor::{storage_traits::*, unique_id, Tensor},
};

use super::{device::QuantizedStorage, Quantize, QuantizedCpu};

use rand::{distributions::Distribution, Rng};
use std::{sync::Arc, vec::Vec};

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> QuantizedCpu<K> {
    #[inline]
    pub(crate) fn try_alloc_zeros(
        &self,
        numel: usize,
    ) -> Result<<Self as DeviceStorage<K::Value>>::Storage, CpuError> {
        self.try_alloc_elem(numel, Default::default())
    }

    #[inline]
    pub(crate) fn try_alloc_elem(
        &self,
        numel: usize,
        elem: K::Value,
    ) -> Result<<Self as DeviceStorage<K::Value>>::Storage, CpuError> {
        #[cfg(feature = "fast-alloc")]
        {
            Ok(QuantizedStorage::from_iter(
                core::iter::repeat_with(|| elem),
                numel,
            ))
        }

        #[cfg(not(feature = "fast-alloc"))]
        {
            let mut data =
                QuantizedStorage::try_with_capacity(numel).map_err(|_| CpuError::OutOfMemory)?;
            data.resize(numel, elem);
            Ok(data)
        }
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> ZerosTensor<K::Value>
    for QuantizedCpu<K>
{
    fn try_zeros_like<S: HasShape>(
        &self,
        src: &S,
    ) -> Result<Tensor<S::Shape, K::Value, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let data = self.try_alloc_zeros(shape.num_elements())?;
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

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> ZeroFillStorage<K::Value>
    for QuantizedCpu<K>
{
    fn try_fill_with_zeros(&self, storage: &mut Self::Storage) -> Result<(), Self::Err> {
        storage.fill(Default::default());
        Ok(())
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> OnesTensor<K::Value>
    for QuantizedCpu<K>
{
    fn try_ones_like<S: HasShape>(
        &self,
        src: &S,
    ) -> Result<Tensor<S::Shape, K::Value, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let data = self.try_alloc_elem(shape.num_elements(), K::Value::ONE)?;
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

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> OneFillStorage<K::Value>
    for QuantizedCpu<K>
{
    fn try_fill_with_ones(&self, storage: &mut Self::Storage) -> Result<(), Self::Err> {
        storage.fill(K::Value::ONE);
        Ok(())
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> SampleTensor<K::Value>
    for QuantizedCpu<K>
{
    fn try_sample_like<S: HasShape, D: Distribution<K::Value>>(
        &self,
        src: &S,
        distr: D,
    ) -> Result<Tensor<S::Shape, K::Value, Self>, Self::Err> {
        let mut tensor = self.try_zeros_like(src)?;
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.cpu.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.cpu.rng.lock();
            let mut iter = Arc::get_mut(&mut tensor.data).unwrap().iter_blocks_mut();
            while let Some(mut block) = iter.next() {
                for v in block.iter_mut() {
                    *v = rng.sample(&distr);
                }
            }
        }
        Ok(tensor)
    }
    fn try_fill_with_distr<D: Distribution<K::Value>>(
        &self,
        storage: &mut Self::Storage,
        distr: D,
    ) -> Result<(), Self::Err> {
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.cpu.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.cpu.rng.lock();
            let mut iter = storage.iter_blocks_mut();
            while let Some(mut block) = iter.next() {
                for v in block.iter_mut() {
                    *v = rng.sample(&distr);
                }
            }
        }
        Ok(())
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> CopySlice<K::Value>
    for QuantizedCpu<K>
{
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, K::Value, Self, T>, src: &[K::Value]) {
        std::sync::Arc::make_mut(&mut dst.data).copy_from_slice(src)
    }
    fn copy_into<S: Shape, T>(src: &Tensor<S, K::Value, Self, T>, dst: &mut [K::Value]) {
        assert_eq!(src.data.len(), dst.len());
        for (i, v) in src.data.iter().enumerate() {
            dst[i] = v;
        }
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> TensorFromVec<K::Value>
    for QuantizedCpu<K>
{
    fn try_tensor_from_vec<S: Shape>(
        &self,
        src: Vec<K::Value>,
        shape: S,
    ) -> Result<Tensor<S, K::Value, Self>, Self::Err> {
        let num_elements = shape.num_elements();

        if src.len() != num_elements {
            Err(CpuError::WrongNumElements)
        } else {
            Ok(Tensor {
                id: unique_id(),
                data: Arc::new(QuantizedStorage::from_slice(&src)),
                shape,
                strides: shape.strides(),
                device: self.clone(),
                tape: Default::default(),
            })
        }
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> TensorToArray<Rank0, K::Value>
    for QuantizedCpu<K>
{
    type Array = K::Value;
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank0, K::Value, Self, T>) -> Self::Array {
        let mut out: Self::Array = Default::default();
        out.clone_from(&tensor.data.get(0).unwrap());
        out
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync, const M: usize>
    TensorToArray<Rank1<M>, K::Value> for QuantizedCpu<K>
{
    type Array = [K::Value; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank1<M>, K::Value, Self, T>) -> Self::Array {
        let mut out: Self::Array = [Default::default(); M];
        let mut iter = tensor.iter();
        for m in 0..M {
            out[m].clone_from(&iter.next().unwrap());
        }
        out
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync, const M: usize, const N: usize>
    TensorToArray<Rank2<M, N>, K::Value> for QuantizedCpu<K>
{
    type Array = [[K::Value; N]; M];
    fn tensor_to_array<T>(&self, tensor: &Tensor<Rank2<M, N>, K::Value, Self, T>) -> Self::Array {
        let mut out: Self::Array = [[Default::default(); N]; M];
        let mut iter = tensor.iter();
        for m in 0..M {
            for n in 0..N {
                out[m][n].clone_from(&iter.next().unwrap());
            }
        }
        out
    }
}

impl<
        K: 'static + Quantize + std::fmt::Debug + Send + Sync,
        const M: usize,
        const N: usize,
        const O: usize,
    > TensorToArray<Rank3<M, N, O>, K::Value> for QuantizedCpu<K>
{
    type Array = [[[K::Value; O]; N]; M];
    fn tensor_to_array<T>(
        &self,
        tensor: &Tensor<Rank3<M, N, O>, K::Value, Self, T>,
    ) -> Self::Array {
        let mut out: Self::Array = [[[Default::default(); O]; N]; M];
        let mut iter = tensor.iter_with_index();
        while let Some((v, [m, n, o])) = iter.next() {
            out[m][n][o].clone_from(&v);
        }
        out
    }
}

impl<
        K: 'static + Quantize + std::fmt::Debug + Send + Sync,
        const M: usize,
        const N: usize,
        const O: usize,
        const P: usize,
    > TensorToArray<Rank4<M, N, O, P>, K::Value> for QuantizedCpu<K>
{
    type Array = [[[[K::Value; P]; O]; N]; M];
    fn tensor_to_array<T>(
        &self,
        tensor: &Tensor<Rank4<M, N, O, P>, K::Value, Self, T>,
    ) -> Self::Array {
        let mut out: Self::Array = [[[[Default::default(); P]; O]; N]; M];
        let mut iter = tensor.iter_with_index();
        while let Some((v, [m, n, o, p])) = iter.next() {
            out[m][n][o][p].clone_from(&v);
        }
        out
    }
}
