use super::structs::*;
use super::traits::HasUniqueId;

impl<Tape> HasUniqueId for Tensor0D<Tape> {
    fn id(&self) -> usize {
        self.id
    }
}

impl<const N: usize, Tape> HasUniqueId for Tensor1D<N, Tape> {
    fn id(&self) -> usize {
        self.id
    }
}

impl<const M: usize, const N: usize, Tape> HasUniqueId for Tensor2D<M, N, Tape> {
    fn id(&self) -> usize {
        self.id
    }
}

impl<const M: usize, const N: usize, const O: usize, Tape> HasUniqueId for Tensor3D<M, N, O, Tape> {
    fn id(&self) -> usize {
        self.id
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, Tape> HasUniqueId
    for Tensor4D<M, N, O, P, Tape>
{
    fn id(&self) -> usize {
        self.id
    }
}
