use super::*;

pub trait Duplicate: Tensor + Sized {
    fn duplicate(self) -> (Self, Self::NoTape);
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> Duplicate for $typename<$($Vs, )* H> {
    fn duplicate(self) -> ($typename<$($Vs, )* H>, $typename<$($Vs, )* NoTape>) {
        let no_tape = $typename {
            id: self.id,
            data: self.data.clone(),
            tape: Default::default(),
        };
        (self, no_tape)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
