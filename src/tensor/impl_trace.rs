use super::*;
use crate::gradients::{NoTape, OwnsTape};

/// Transforms a [NoTape] tensor to an [OwnsTape] tensor by cloning.
/// Clones `t` using [Tensor::duplicate()] (to preserve id), and then
/// inserts [OwnsTape] as the tape.
///
/// See [traced()] for version that takes ownership of `t`.
pub fn trace<T: Tensor<Tape = OwnsTape>>(t: &T::NoTape) -> T {
    traced(t.duplicate())
}

/// Transforms a [NoTape] tensor to an [OwnsTape] by directly inserting a
/// new [OwnsTape] into `t`.
///
/// See [trace()] for version that copies `t`.
pub fn traced<T: Tensor<Tape = OwnsTape>>(t: T::NoTape) -> T {
    t.put_tape(OwnsTape::default())
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> $typename<$($Vs, )* NoTape> {
    /// Clones `self` and returns a copy with [OwnsTape] as the [crate::gradients::Tape].
    ///
    /// See `traced` for a version that takes ownership of the tensor.
    pub fn trace(&self) -> $typename<$($Vs, )* OwnsTape> {
        trace(self)
    }

    /// Takes ownership of `self` and inserts [OwnsTape] as the [crate::gradients::Tape].
    pub fn traced(self) -> $typename<$($Vs, )* OwnsTape> {
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
