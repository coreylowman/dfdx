use super::*;
use crate::gradients::{NoneTape, OwnedTape};

/// Transforms a [NoTape] tensor to an [OwnedTape] tensor by cloning.
/// Clones `t` using [Tensor::duplicate()] (to preserve id), and then
/// inserts [OwnedTape] as the tape.
///
/// See [traced()] for version that takes ownership of `t`.
pub fn trace<T: Tensor<Tape = OwnedTape>>(t: &T::NoTape) -> T {
    traced(t.duplicate())
}

/// Transforms a [NoTape] tensor to an [OwnedTape] by directly inserting a
/// new [OwnedTape] into `t`.
///
/// See [trace()] for version that copies `t`.
pub fn traced<T: Tensor<Tape = OwnedTape>>(t: T::NoTape) -> T {
    t.put_tape(OwnedTape::default())
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> $typename<$($Vs, )* NoneTape> {
    /// Clones `self` and returns a copy with [OwnedTape] as the [crate::gradients::Tape].
    ///
    /// See `traced` for a version that takes ownership of the tensor.
    pub fn trace(&self) -> $typename<$($Vs, )* OwnedTape> {
        trace(self)
    }

    /// Takes ownership of `self` and inserts [OwnedTape] as the [crate::gradients::Tape].
    pub fn traced(self) -> $typename<$($Vs, )* OwnedTape> {
        traced(self)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
