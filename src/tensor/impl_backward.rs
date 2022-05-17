use super::*;
use crate::{
    gradients::Gradients,
    prelude::{Cpu, FillElements},
};

pub fn backward<T: Tensor<TapeHolder = WithTape>>(t: T) -> Gradients
where
    Cpu: FillElements<T::Array>,
{
    let (t, tape_holder) = t.split_tape_holder();
    tape_holder.0.backward(&t)
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
