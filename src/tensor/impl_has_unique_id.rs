use super::structs::*;
use crate::unique_id::{HasUniqueId, UniqueId};

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H> HasUniqueId for $typename<$($Vs, )* H> {
    fn id(&self) -> &UniqueId {
        &self.id
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
tensor_impl!(Tensor5D, [M, N, O, P, Q]);
tensor_impl!(Tensor6D, [M, N, O, P, Q, R]);
