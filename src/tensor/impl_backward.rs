use super::*;
use crate::{devices::FillElements, gradients::Gradients};

pub fn backward<T: Tensor<TapeHolder = WithTape>>(t: T) -> Gradients
where
    T::Device: FillElements<T::Array>,
{
    let (t, mut tape_holder) = t.split_tape_holder();
    tape_holder.add_operation(move |tape| {
        T::Device::fill(tape.mut_gradient(&t), &mut |v| *v = 1.0);
    });
    tape_holder.0.execute()
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> $typename<$($Vs, )* WithTape> {
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
