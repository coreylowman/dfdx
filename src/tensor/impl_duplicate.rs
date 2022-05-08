use super::*;

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> $typename<$($Vs, )* H> {
    pub fn duplicate(&self) -> $typename<$($Vs, )* NoTape> {
        $typename {
            id: self.id,
            data: self.data.clone(),
            tape: Default::default(),
        }
    }
}

impl<$(const $Vs: usize, )*> Clone for $typename<$($Vs, )* NoTape> {
    fn clone(&self) -> Self {
        $typename::new(self.data.clone())
    }
}


    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
