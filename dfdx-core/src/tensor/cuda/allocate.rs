#![allow(clippy::needless_range_loop)]

use crate::{
    shapes::*,
    tensor::{masks::triangle_mask, storage_traits::*, unique_id, Cpu, Error, NoneTape, Tensor},
};

use super::{device::CachableCudaSlice, Cuda};

use cudarc::driver::{CudaSlice, DeviceSlice};
use rand::Rng;
use std::{sync::Arc, vec::Vec};

impl Cuda {
    fn tensor_from_host_buf<S: Shape, E: Unit>(
        &self,
        shape: S,
        buf: Vec<E>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let mut slice = unsafe { self.alloc_empty(buf.len()) }?;
        self.dev.htod_copy_into(buf, &mut slice)?;
        Ok(self.build_tensor(shape, shape.strides(), slice))
    }

    pub(crate) fn build_tensor<S: Shape, E: Unit>(
        &self,
        shape: S,
        strides: S::Concrete,
        slice: CudaSlice<E>,
    ) -> Tensor<S, E, Self> {
        let data = CachableCudaSlice {
            data: slice,
            cache: self.cache.clone(),
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

impl<E: Unit> ZerosTensor<E> for Cuda {
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let strides = shape.strides();
        let mut data = unsafe { self.alloc_empty(shape.num_elements()) }?;
        self.dev.memset_zeros(&mut data)?;
        Ok(self.build_tensor(shape, strides, data))
    }
}

impl<E: Unit> ZeroFillStorage<E> for Cuda {
    fn try_fill_with_zeros(&self, storage: &mut Self::Vec) -> Result<(), Error> {
        self.dev.memset_zeros(&mut storage.data)?;
        Ok(())
    }
}

impl<E: Unit> WithStorage<E> for Cuda {
    /// View a copy of the values by each element (not in-place).
    fn try_element_view<F: FnMut(&E)>(&self, storage: &Self::Vec, mut f: F) -> Result<(), Error> {
        let v = self.dev.dtoh_sync_copy(storage)?;
        for e in v.iter() {
            f(e);
        }
        Ok(())
    }
    /// View a copy of the values by a [Vec] (not in-place).
    fn try_view<F: FnMut(&[E])>(&self, storage: &Self::Vec, mut f: F) -> Result<(), Error> {
        let v = self.dev.dtoh_sync_copy(storage)?;
        f(v.as_slice());
        Ok(())
    }
    /// Mutates a copy of the values by each element (not in-place).
    /// Then the values in Cuda memory are replaced by the changed values.
    fn try_element_map<F: FnMut(E) -> E>(
        &self,
        storage: &mut Self::Vec,
        mut f: F,
    ) -> Result<(), Error> {
        let mut v = self.dev.dtoh_sync_copy(storage)?;
        for e in v.iter_mut() {
            let fe = (&mut f)(*e);
            *e = fe;
        }
        self.dev.htod_copy_into(v, storage)?;
        Ok(())
    }
    /// Mutates a copy of the values (not in-place).
    ///
    /// If `Some` is returned, the values in Cuda memory are replaced by the changed values.  
    /// Otherwise if `None` is returned, the values in Cuda memory are left intact.
    fn try_map<F: FnMut(Vec<E>) -> Option<Vec<E>>>(
        &self,
        storage: &mut Self::Vec,
        mut f: F,
    ) -> Result<(), Error> {
        let v = self.dev.dtoh_sync_copy(storage)?;
        if let Some(fv) = (&mut f)(v) {
            self.dev.htod_copy_into(fv, storage)?;
        }
        Ok(())
    }
}

impl<E: Unit> OnesTensor<E> for Cuda
where
    Cpu: OnesTensor<E>,
{
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Error> {
        let shape = *src.shape();
        let buf = std::vec![E::ONE; shape.num_elements()];
        self.tensor_from_host_buf(shape, buf)
    }
}

impl<E: Unit> TriangleTensor<E> for Cuda
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
        let mut data = std::vec![val; shape.num_elements()];
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
        let mut data = std::vec![val; shape.num_elements()];
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(&mut data, &shape, false, offset);
        self.tensor_from_host_buf(shape, data)
    }
}

impl<E: Unit> OneFillStorage<E> for Cuda {
    fn try_fill_with_ones(&self, storage: &mut Self::Vec) -> Result<(), Error> {
        self.dev
            .htod_copy_into(std::vec![E::ONE; storage.len()], &mut storage.data)?;
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
    fn try_fill_with_distr<D: rand_distr::Distribution<E>>(
        &self,
        storage: &mut Self::Vec,
        distr: D,
    ) -> Result<(), Error> {
        let mut buf = Vec::with_capacity(storage.len());
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.cpu.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.cpu.rng.lock();
            buf.resize_with(storage.len(), || rng.sample(&distr));
        }
        self.dev.htod_copy_into(buf, &mut storage.data)?;
        Ok(())
    }
}

impl<E: Unit> CopySlice<E> for Cuda {
    fn copy_from<S: Shape, T>(dst: &mut Tensor<S, E, Self, T>, src: &[E]) {
        assert_eq!(
            dst.data.len(),
            src.len(),
            "Slices must have same number of elements as *physical* Storage<E> of tensors."
        );
        let storage = Arc::make_mut(&mut dst.data);
        dst.device
            .dev
            .htod_sync_copy_into(src, &mut storage.data)
            .unwrap();
    }
    fn copy_into<S: Shape, T>(src: &Tensor<S, E, Self, T>, dst: &mut [E]) {
        assert_eq!(
            src.data.len(),
            dst.len(),
            "Slices must have same number of elements as *physical* Storage<E> of tensors."
        );
        let storage: &Self::Vec = src.data.as_ref();
        src.device
            .dev
            .dtoh_sync_copy_into(&storage.data, dst)
            .unwrap();
    }
}

impl<E: Unit> TensorFromVec<E> for Cuda {
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

impl<S: Shape, E: Unit> TensorToArray<S, E> for Cuda
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
