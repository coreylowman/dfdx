use crate::{
    prelude::{HasErr, HasShape},
    shapes::Shape,
    tensor::Storage,
};

use super::{storage_traits::AllocGrad, GhostTensor, Tensor, UniqueId};

/// Contains everything that comprises a tensor, except possibly for the actual data. This really
/// exists to unify handling of [Tensor] and [GhostTensor].
///
/// *If it looks like a tensor and barks like a tensor, then pet it like a tensor.*
pub trait Tensorlike<S: Shape, E, D: Storage<E>>:
    AllocGrad<Gradient = D::Vec> + HasErr<Err = D::Err> + HasShape<Shape = S>
{
    fn id(&self) -> UniqueId;
    fn len(&self) -> usize;
    fn strides(&self) -> S::Concrete;
    fn dev(&self) -> &D;
    fn data(&self) -> Option<&D::Vec>;
}

impl<S: Shape, E, D: Storage<E>, T> Tensorlike<S, E, D> for Tensor<S, E, D, T> {
    fn id(&self) -> UniqueId {
        self.id
    }

    fn len(&self) -> usize {
        self.device.len(&self.data)
    }

    fn strides(&self) -> S::Concrete {
        self.strides
    }

    fn dev(&self) -> &D {
        &self.device
    }

    fn data(&self) -> Option<&D::Vec> {
        Some(self.data.as_ref())
    }
}

impl<S: Shape, E, D: Storage<E>> Tensorlike<S, E, D> for GhostTensor<S, E, D> {
    fn id(&self) -> UniqueId {
        self.id
    }

    fn len(&self) -> usize {
        self.len
    }

    fn strides(&self) -> S::Concrete {
        self.strides
    }

    fn dev(&self) -> &D {
        &self.dev
    }

    fn data(&self) -> Option<&D::Vec> {
        None
    }
}
