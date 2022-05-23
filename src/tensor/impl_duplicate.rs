use super::*;

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H> $typename<$($Vs, )* H> {
    /// Clones `self`, but returns something with [NoTape] regardless of whether `self` holds the tape.
    pub fn duplicate(&self) -> $typename<$($Vs, )* NoTape> {
        $typename {
            id: self.id,
            data: self.data.clone(),
            tape: Default::default(),
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
