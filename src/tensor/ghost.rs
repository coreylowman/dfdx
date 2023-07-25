use crate::{shapes::*, tensor::*};

/// Holds all the information a [Tensor] does, except without
/// holding a reference to the data storage.
///
/// This can held reduce memory usage by decreasing reference
/// count on tensor data, meaning data can be re-used more.
pub struct GhostTensor<S: Shape, E, D: Storage<E>> {
    pub(crate) id: UniqueId,
    pub(crate) len: usize,
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
    pub(crate) dev: D,
    marker: std::marker::PhantomData<E>,
}

impl<S: Shape, E, D: Storage<E>, T> Tensor<S, E, D, T> {
    /// Creates a ghost tensor that doesn't hold a reference
    /// to the tensor's data.
    pub fn ghost(&self) -> GhostTensor<S, E, D> {
        GhostTensor {
            id: self.id,
            len: self.device.len(&self.data),
            shape: self.shape,
            strides: self.strides,
            dev: self.device.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<S: Shape, E, D: Storage<E>> Clone for GhostTensor<S, E, D> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            len: self.len,
            shape: self.shape,
            strides: self.strides,
            dev: self.dev.clone(),
            marker: self.marker,
        }
    }
}

impl<S: Shape, E, D: Storage<E>> super::storage_traits::HasErr for GhostTensor<S, E, D> {
    type Err = D::Err;
}

impl<S: Shape, E, D: Storage<E>> HasShape for GhostTensor<S, E, D> {
    type WithShape<New: Shape> = GhostTensor<New, E, D>;
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }
}

impl<S: Shape, E, D: Storage<E>> super::storage_traits::AllocGrad for GhostTensor<S, E, D> {
    type Gradient = D::Vec;
    fn try_alloc_grad(&self) -> Result<Self::Gradient, D::Err> {
        self.dev.try_alloc_len(self.len)
    }
}
