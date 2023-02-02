mod cpu_kernels;

#[cfg(feature = "cuda")]
mod cuda_kernels;

use crate::{
    prelude::{OnesTensor, Tensor, ZerosTensor},
    shapes::*,
    tensor::DeviceStorage,
};

use std::ops::{BitAnd, BitOr, BitXor, Not};

use super::Device;

pub trait BooleanKernel: DeviceStorage + OnesTensor<bool> + ZerosTensor<bool> {
    fn not<S: Shape>(
        &self,
        inp: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err>;

    fn and<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err>;

    fn or<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err>;

    fn xor<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: &Self::Storage<S, bool>,
    ) -> Result<Self::Storage<S, bool>, Self::Err>;
}

fn scalar_and<D: BooleanKernel, S: Shape>(
    lhs: &Tensor<S, bool, D>,
    rhs: bool,
) -> Result<Tensor<S, bool, D>, D::Err> {
    if rhs {
        Ok(lhs.clone())
    } else {
        lhs.device.try_zeros_like(lhs)
    }
}

fn scalar_or<D: BooleanKernel, S: Shape>(
    lhs: &Tensor<S, bool, D>,
    rhs: bool,
) -> Result<Tensor<S, bool, D>, D::Err> {
    if rhs {
        lhs.device.try_ones_like(lhs)
    } else {
        Ok(lhs.clone())
    }
}

fn scalar_xor<D: BooleanKernel, S: Shape>(
    lhs: &Tensor<S, bool, D>,
    rhs: bool,
) -> Result<Tensor<S, bool, D>, D::Err> {
    if rhs {
        Ok(lhs.device.upgrade(lhs.device.not(&lhs.storage)?))
    } else {
        Ok(lhs.clone())
    }
}

impl<S: Shape, D: BooleanKernel> Not for Tensor<S, bool, D> {
    type Output = Self;

    fn not(self) -> Self {
        !&self
    }
}

impl<S: Shape, D: BooleanKernel> Not for &Tensor<S, bool, D> {
    type Output = Tensor<S, bool, D>;

    fn not(self) -> Self::Output {
        self.device.upgrade(self.device.not(&self.storage).unwrap())
    }
}

macro_rules! boolean_op_impl {
    ($op:ident, $op_method:ident, $binary_kernel_method:ident, $scalar_function:ident) => {
        impl<S: Shape, D: BooleanKernel> $op for Tensor<S, bool, D> {
            type Output = Self;

            fn $op_method(self, rhs: Self) -> Self {
                assert_eq!(self.shape(), rhs.shape());
                self.device.upgrade(
                    self.device
                        .$binary_kernel_method(&self.storage, &rhs.storage)
                        .unwrap(),
                )
            }
        }

        impl<S: Shape, D: BooleanKernel> $op for &Tensor<S, bool, D> {
            type Output = Tensor<S, bool, D>;

            fn $op_method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                self.device.upgrade(
                    self.device
                        .$binary_kernel_method(&self.storage, &rhs.storage)
                        .unwrap(),
                )
            }
        }

        impl<S: Shape, D: BooleanKernel> $op<bool> for Tensor<S, bool, D> {
            type Output = Self;

            fn $op_method(self, rhs: bool) -> Self {
                $scalar_function(&self, rhs).unwrap()
            }
        }

        impl<S: Shape, D: BooleanKernel> $op<bool> for &Tensor<S, bool, D> {
            type Output = Tensor<S, bool, D>;

            fn $op_method(self, rhs: bool) -> Self::Output {
                $scalar_function(self, rhs).unwrap()
            }
        }
    };
}

boolean_op_impl!(BitAnd, bitand, and, scalar_and);
boolean_op_impl!(BitOr, bitor, or, scalar_or);
boolean_op_impl!(BitXor, bitxor, xor, scalar_xor);

/// Inverts each value in a boolean tensor.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([false, true, false]);
///
/// // Can take either a reference or an owned value
/// let r1 = !&a;
/// let r2 = !a;
/// assert_eq!(r1.array(), [true, false, true]);
/// assert_eq!(r2.array(), [true, false, true]);
/// ```
pub fn bool_not<S: Shape, E: Dtype, D: Device<E>>(inp: &Tensor<S, bool, D>) -> Tensor<S, bool, D> {
    !inp
}

/// Element wise and scalar boolean 'and'.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([false, false, true, true]);
/// let b = dev.tensor([false, true, false, true]);
///
/// // Can take either references or owned values
/// let r1 = &a & &b;
/// let r2 = a & b;
/// assert_eq!(r1.array(), [false, false, false, true]);
/// assert_eq!(r2.array(), [false, false, false, true]);
/// ```
///
/// And-ing with a scalar:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([false, true, false]);
///
/// let r1 = &a & true;
/// let r2 = &a & false;
///
/// assert_eq!(r1.array(), a.array());
/// assert_eq!(r2.array(), [false; 3]);
/// ```
pub fn bool_and<S: Shape, E: Dtype, D: Device<E>>(
    lhs: &Tensor<S, bool, D>,
    rhs: &Tensor<S, bool, D>,
) -> Tensor<S, bool, D> {
    lhs & rhs
}

/// Element wise and scalar boolean 'or'.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([false, false, true, true]);
/// let b = dev.tensor([false, true, false, true]);
///
/// // Can take either references or owned values
/// let r1 = &a | &b;
/// let r2 = a | b;
/// assert_eq!(r1.array(), [false, true, true, true]);
/// assert_eq!(r2.array(), [false, true, true, true]);
/// ```
///
/// And-ing with a scalar:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([false, true, false]);
///
/// let r1 = &a | true;
/// let r2 = &a | false;
///
/// assert_eq!(r1.array(), [true; 3]);
/// assert_eq!(r2.array(), a.array());
/// ```
pub fn bool_or<S: Shape, E: Dtype, D: Device<E>>(
    lhs: &Tensor<S, bool, D>,
    rhs: &Tensor<S, bool, D>,
) -> Tensor<S, bool, D> {
    lhs | rhs
}

/// Element wise and scalar boolean 'xor'.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([false, false, true, true]);
/// let b = dev.tensor([false, true, false, true]);
///
/// // Can take either references or owned values
/// let r1 = &a ^ &b;
/// let r2 = a ^ b;
/// assert_eq!(r1.array(), [false, true, true, false]);
/// assert_eq!(r2.array(), [false, true, true, false]);
/// ```
///
/// And-ing with a scalar:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([false, true, false]);
///
/// let r1 = &a ^ true;
/// let r2 = &a ^ false;
///
/// assert_eq!(r1.array(), (!&a).array());
/// assert_eq!(r2.array(), a.array());
/// ```
pub fn bool_xor<S: Shape, E: Dtype, D: Device<E>>(
    lhs: &Tensor<S, bool, D>,
    rhs: &Tensor<S, bool, D>,
) -> Tensor<S, bool, D> {
    lhs ^ rhs
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::*};

    const TRUTH_TABLE_1: [bool; 4] = [false, false, true, true];
    const TRUTH_TABLE_2: [bool; 4] = [false, true, false, true];

    #[test]
    fn test_boolean_not() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([TRUTH_TABLE_1, TRUTH_TABLE_2]);

        let r = !a;
        assert_eq!(
            r.array(),
            [[true, true, false, false], [true, false, true, false]]
        );
    }

    #[test]
    fn test_boolean_and() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([TRUTH_TABLE_1, TRUTH_TABLE_2]);
        let b = dev.tensor([TRUTH_TABLE_2, TRUTH_TABLE_1]);

        let r1 = &a & &b;
        let r2 = &a & true;
        let r3 = &a & false;
        assert_eq!(r1.array(), [[false, false, false, true]; 2]);
        assert_eq!(r2.array(), a.array());
        assert_eq!(r3.array(), dev.zeros_like(&a).array());
    }

    #[test]
    fn test_boolean_or() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([TRUTH_TABLE_1, TRUTH_TABLE_2]);
        let b = dev.tensor([TRUTH_TABLE_2, TRUTH_TABLE_1]);

        let r1 = &a | &b;
        let r2 = &a | true;
        let r3 = &a | false;
        assert_eq!(r1.array(), [[false, true, true, true]; 2]);
        assert_eq!(r2.array(), dev.ones_like(&a).array());
        assert_eq!(r3.array(), a.array());
    }

    #[test]
    fn test_boolean_xor() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([TRUTH_TABLE_1, TRUTH_TABLE_2]);
        let b = dev.tensor([TRUTH_TABLE_2, TRUTH_TABLE_1]);

        let r1 = &a ^ &b;
        let r2 = &a ^ true;
        let r3 = &a ^ false;
        assert_eq!(r1.array(), [[false, true, true, false]; 2]);
        assert_eq!(r2.array(), (!&a).array());
        assert_eq!(r3.array(), a.array());
    }
}
