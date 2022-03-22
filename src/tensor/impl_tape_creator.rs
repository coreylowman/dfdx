use super::*;

pub trait TapeCreator: Tensor {
    fn with_tape(&self) -> Self::WithTape;
}

impl TapeCreator for Tensor0D<NoTape> {
    fn with_tape(&self) -> Self::WithTape {
        Self::WithTape {
            id: self.id,
            data: self.data.clone(),
            tape: WithTape::default(),
        }
    }
}

impl<const N: usize> TapeCreator for Tensor1D<N, NoTape> {
    fn with_tape(&self) -> Self::WithTape {
        Self::WithTape {
            id: self.id,
            data: self.data.clone(),
            tape: WithTape::default(),
        }
    }
}

impl<const M: usize, const N: usize> TapeCreator for Tensor2D<M, N, NoTape> {
    fn with_tape(&self) -> Self::WithTape {
        Self::WithTape {
            id: self.id,
            data: self.data.clone(),
            tape: WithTape::default(),
        }
    }
}

impl<const M: usize, const N: usize, const O: usize> TapeCreator for Tensor3D<M, N, O, NoTape> {
    fn with_tape(&self) -> Self::WithTape {
        Self::WithTape {
            id: self.id,
            data: self.data.clone(),
            tape: WithTape::default(),
        }
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize> TapeCreator
    for Tensor4D<M, N, O, P, NoTape>
{
    fn with_tape(&self) -> Self::WithTape {
        Self::WithTape {
            id: self.id,
            data: self.data.clone(),
            tape: WithTape::default(),
        }
    }
}
