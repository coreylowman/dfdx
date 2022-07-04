use super::*;
use crate::gradients::{NoTape, OwnsTape};

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> $typename<$($Vs, )* NoTape> {
    /// Clones `self` and returns a copy with [OwnsTape] as the [crate::gradients::Tape].
    ///
    /// See `traced` for a version that takes ownership of the tensor.
    pub fn trace(&self) -> $typename<$($Vs, )* OwnsTape> {
        $typename {
            id: self.id,
            data: self.data.clone(),
            tape: OwnsTape::default(),
        }
    }

    /// Takes ownership of `self` and inserts [OwnsTape] as the [crate::gradients::Tape].
    pub fn traced(self) -> $typename<$($Vs, )* OwnsTape> {
        $typename {
            id: self.id,
            data: self.data,
            tape: OwnsTape::default(),
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
