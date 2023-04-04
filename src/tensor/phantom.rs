use crate::{shapes::*, tensor::*};

pub(crate) struct PhantomTensor<D: DeviceStorage> {
    pub(crate) id: UniqueId,
    pub(crate) len: usize,
    pub(crate) dev: D,
}

impl<S: Shape, E: Unit, D: DeviceStorage, T> Tensor<S, E, D, T> {
    pub(crate) fn phantom(&self) -> PhantomTensor<D> {
        PhantomTensor {
            id: self.id,
            len: self.device.len(&self.data),
            dev: self.device.clone(),
        }
    }
}
