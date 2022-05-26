use super::*;
use crate::{
    devices::FillElements,
    gradients::{Gradients, OwnsTape, Tape},
};

/// Runs backprop algorithm with all operations contained in the tape that `t` has.
///
/// This function takes ownership of `t` and returns [Gradients].
///
/// Note that `t` is required to have [OwnsTape], which means it currently owns the [GradientTape].
pub fn backward<T: Tensor<Dtype = f32, Tape = OwnsTape>>(t: T) -> Gradients {
    let (t, mut tape) = t.split_tape();
    tape.add_backward_op(move |grads| {
        T::Device::fill(grads.mut_gradient(&t), &mut || num_traits::One::one());
    });
    tape.0.execute()
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> $typename<$($Vs, )* OwnsTape> {
    #[doc="Calls [backward()] on `self`"]
    pub fn backward(self) -> Gradients {
        backward(self)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
