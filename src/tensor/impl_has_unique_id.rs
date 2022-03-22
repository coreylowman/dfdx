use super::structs::*;

pub trait HasUniqueId {
    fn id(&self) -> usize;
}

impl<H> HasUniqueId for Tensor0D<H> {
    fn id(&self) -> usize {
        self.id
    }
}

impl<const N: usize, H> HasUniqueId for Tensor1D<N, H> {
    fn id(&self) -> usize {
        self.id
    }
}

impl<const M: usize, const N: usize, H> HasUniqueId for Tensor2D<M, N, H> {
    fn id(&self) -> usize {
        self.id
    }
}

impl<const M: usize, const N: usize, const O: usize, H> HasUniqueId for Tensor3D<M, N, O, H> {
    fn id(&self) -> usize {
        self.id
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H> HasUniqueId
    for Tensor4D<M, N, O, P, H>
{
    fn id(&self) -> usize {
        self.id
    }
}
