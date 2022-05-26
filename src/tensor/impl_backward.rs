use super::*;
use crate::{devices::FillElements, gradients::Gradients};

/// Runs backprop algorithm with all operations contained in the tape that `t` has.
///
/// This function takes ownership of `t` and returns [Gradients].
///
/// Note that `t` is required to have [WithTape], which means it currently owns the [GradientTape].
pub fn backward<T: Tensor<Dtype = f32, TapeHolder = WithTape>>(t: T) -> Gradients {
    let (t, mut with_tape) = t.split_tape_holder();
    with_tape.add_operation(move |tape| {
        T::Device::fill(tape.mut_gradient(&t), &mut || num_traits::One::one());
    });
    with_tape.0.execute()
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> $typename<$($Vs, )* WithTape> {
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
