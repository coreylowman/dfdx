mod cpu_kernels;

#[cfg(feature = "cuda")]
mod cuda_kernels;

use crate::{
    shapes::*,
    tensor::DeviceStorage, prelude::{OnesTensor, ZerosTensor, Tensor},
};

use std::ops::{BitAnd, BitOr, BitXor, Not};

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

    fn scalar_and<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: bool,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        if rhs {
            Ok(lhs.clone())
        } else {
            Ok(self.try_zeros_like(lhs)?.storage)
        }
    }

    fn scalar_or<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: bool,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        if rhs {
            Ok(self.try_ones_like(lhs)?.storage)
        } else {
            Ok(lhs.clone())
        }
    }

    fn scalar_xor<S: Shape>(
        &self,
        lhs: &Self::Storage<S, bool>,
        rhs: bool,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        if rhs {
            self.not(lhs)
        } else {
            Ok(lhs.clone())
        }
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
    ($op:ident, $method1:ident, $method2:ident, $method3:ident) => {
        impl<S: Shape, D: BooleanKernel> $op for Tensor<S, bool, D> {
            type Output = Self;

            fn $method1(self, rhs: Self) -> Self {
                self.device.upgrade(self.device.$method2(&self.storage, &rhs.storage).unwrap())
            }
        }

        impl<S: Shape, D: BooleanKernel> $op for &Tensor<S, bool, D> {
            type Output = Tensor<S, bool, D>;

            fn $method1(self, rhs: Self) -> Self::Output {
                self.device.upgrade(self.device.$method2(&self.storage, &rhs.storage).unwrap())
            }
        }

        impl<S: Shape, D: BooleanKernel> $op<bool> for Tensor<S, bool, D> {
            type Output = Self;

            fn $method1(self, rhs: bool) -> Self {
                self.device.upgrade(self.device.$method3(&self.storage, rhs).unwrap())
            }
        }

        impl<S: Shape, D: BooleanKernel> $op<bool> for &Tensor<S, bool, D> {
            type Output = Tensor<S, bool, D>;

            fn $method1(self, rhs: bool) -> Self::Output {
                self.device.upgrade(self.device.$method3(&self.storage, rhs).unwrap())
            }
        }
    }
}

boolean_op_impl!(BitAnd, bitand, and, scalar_and);
boolean_op_impl!(BitOr, bitor, or, scalar_or);
boolean_op_impl!(BitXor, bitxor, xor, scalar_xor);

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
        assert_eq!(
            r1.array(),
            [[false, false, false, true]; 2]
        );
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
        assert_eq!(
            r1.array(),
            [[false, true, true, true]; 2]
        );
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
        assert_eq!(
            r1.array(),
            [[false, true, true, false]; 2]
        );
        assert_eq!(r2.array(), (!&a).array());
        assert_eq!(r3.array(), a.array());
    }
}
