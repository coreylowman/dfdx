use super::{Tensor0D, Tensor1D, Tensor2D, Tensor3D, Tensor4D, TensorCreator};
use std::boxed::Box;

/// Creates a tensor using the data based in. The return type is based
/// on the data you pass in. See [IntoTensor] for implementations.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let _/*: Tensor0D*/ = tensor(0.0);
/// let _/*: Tensor1D<3>*/ = tensor([0.0, 1.0, 2.0]);
/// let _/*: Tensor2D<2, 3>*/ = tensor([[0.0; 3]; 2]);
/// ```
pub fn tensor<T: IntoTensor>(data: T) -> T::Tensor {
    data.into_tensor()
}

/// Enables converting this value into a Tensor. See [tensor()].
pub trait IntoTensor: Sized {
    /// The type of tensor that this value would be converted into.
    type Tensor: TensorCreator;

    /// Convert this value into a tensor.
    fn into_tensor(self) -> Self::Tensor;
}

macro_rules! impl_into_tensor {
    ($ArrTy:ty, $TensorTy:ty, {$($Dims:tt),*}) => {
impl<$(const $Dims: usize, )*> IntoTensor for $ArrTy {
    type Tensor = $TensorTy;
    fn into_tensor(self) -> Self::Tensor {
        TensorCreator::new(self)
    }
}
impl<$(const $Dims: usize, )*> IntoTensor for Box<$ArrTy> {
    type Tensor = $TensorTy;
    fn into_tensor(self) -> Self::Tensor {
        TensorCreator::new_boxed(self)
    }
}
    };
}

impl_into_tensor!(f32, Tensor0D, {});
impl_into_tensor!([f32; M], Tensor1D<M>, { M });
impl_into_tensor!([[f32; N]; M], Tensor2D<M, N>, {M, N});
impl_into_tensor!([[[f32; O]; N]; M], Tensor3D<M, N, O>, {M, N, O});
impl_into_tensor!([[[[f32; P]; O]; N]; M], Tensor4D<M, N, O, P>, {M, N, O, P});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_0d_into_tensor() {
        let arr = 0.0;
        let a = tensor(arr);
        assert_eq!(a.data(), &arr);
        let a = tensor(Box::new(arr));
        assert_eq!(a.data(), &arr);
    }

    #[test]
    fn test_1d_into_tensor() {
        let arr = [0.0, 1.0, 2.0];
        let a = tensor(arr);
        assert_eq!(a.data(), &arr);
        let a = tensor(Box::new(arr));
        assert_eq!(a.data(), &arr);
    }

    #[test]
    fn test_2d_into_tensor() {
        let arr = [[0.0, 1.0, 2.0], [-1.0, -2.0, -3.0]];
        let a = tensor(arr);
        assert_eq!(a.data(), &arr);
        let a = tensor(Box::new(arr));
        assert_eq!(a.data(), &arr);
    }

    #[test]
    fn test_3d_into_tensor() {
        let arr = [[[0.0, 1.0, 2.0], [-1.0, -2.0, -3.0]]];
        let a = tensor(arr);
        assert_eq!(a.data(), &arr);
        let a = tensor(Box::new(arr));
        assert_eq!(a.data(), &arr);
    }

    #[test]
    fn test_4d_into_tensor() {
        let arr = [[[[0.0, 1.0, 2.0], [-1.0, -2.0, -3.0]]]; 4];
        let a = tensor(arr);
        assert_eq!(a.data(), &arr);
        let a = tensor(Box::new(arr));
        assert_eq!(a.data(), &arr);
    }
}
