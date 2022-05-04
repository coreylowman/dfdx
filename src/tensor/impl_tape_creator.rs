use super::*;

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> $typename<$($Vs, )* NoTape> {
    pub fn with_tape(&self) -> $typename<$($Vs, )* WithTape> {
        $typename {
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
