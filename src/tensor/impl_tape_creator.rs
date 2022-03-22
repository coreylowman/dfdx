use super::*;

pub trait TapeCreator: Tensor {
    fn with_tape(&self) -> Self::WithTape;
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> TapeCreator for $typename<$($Vs, )* NoTape> {
    fn with_tape(&self) -> Self::WithTape {
        Self::WithTape {
            id: self.id,
            data: self.data.clone(),
            tape: WithTape::default(),
        }
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
