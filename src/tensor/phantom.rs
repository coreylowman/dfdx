use crate::{shapes::*, tensor::*};

pub(crate) struct GhostTensor<E: Unit, D: DeviceStorage> {
    pub(crate) id: UniqueId,
    pub(crate) len: usize,
    pub(crate) dev: D,
    marker: std::marker::PhantomData<E>,
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> Tensor<S, E, D, T> {
    pub(crate) fn ghost(&self) -> GhostTensor<E, D> {
        GhostTensor {
            id: self.id,
            len: self.device.len(&self.data),
            dev: self.device.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<E: Unit, D: DeviceStorage> super::storage_traits::HasErr for GhostTensor<E, D> {
    type Err = D::Err;
}

impl<E: Unit, D: DeviceStorage> super::storage_traits::AllocGrad for GhostTensor<E, D> {
    type Gradient = D::Vec<E>;
    fn try_alloc_grad(&self) -> Result<Self::Gradient, D::Err> {
        self.dev.try_alloc_len(self.len)
    }
}
