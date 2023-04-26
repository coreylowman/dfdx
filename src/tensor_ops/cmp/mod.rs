use crate::{
    shapes::{HasShape, Shape, Unit},
    tensor::{DeviceStorage, HasErr, NoneTape, Tape, Tensor},
};

mod cpu_kernels;
#[cfg(feature = "cuda")]
mod cuda_kernels;

pub trait CmpKernel<Op, E: Unit>: DeviceStorage {
    fn forward<S: Shape, T>(
        &self,
        lhs: &Tensor<S, E, Self, T>,
        rhs: &Tensor<S, E, Self, T>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err>;
}

fn try_cmp_op<Op, S: Shape, E: Unit, D: CmpKernel<Op, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
    assert_eq!(lhs.shape(), rhs.shape());
    lhs.device.forward(lhs, rhs)
}

pub trait ScalarCmpKernel<Op, E: Unit>: DeviceStorage {
    fn forward<S: Shape, T>(
        &self,
        tensor: &Tensor<S, E, Self, T>,
        scalar: E,
    ) -> Result<Tensor<S, bool, Self>, Self::Err>;
}

fn try_scalar_cmp_op<Op, S: Shape, E: Unit, D: ScalarCmpKernel<Op, E>, T: Tape<E, D>>(
    tensor: &Tensor<S, E, D, T>,
    scalar: E,
) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
    tensor.device.forward(tensor, scalar)
}

pub enum EqKernelOp {}
pub enum NeKernelOp {}
pub enum GtKernelOp {}
pub enum GeKernelOp {}
pub enum LtKernelOp {}
pub enum LeKernelOp {}

/// Element-wise equality comparison. `==`
///
/// Examples:
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([1.2, 3.4, -0.5, -6.7]);
/// let b = dev.tensor([1.2, 0.0, 3.14, -6.7]);
/// let r = a.eq(&b);
/// assert_eq!(r.array(), [true, false, false, true]);
/// ```
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([-2, -1, 0, 1, 2]);
/// let r = a.eq(1);
/// assert_eq!(r.array(), [false, false, false, true, false]);
/// ```
pub fn eq<S: Shape, E: Unit, D: CmpKernel<EqKernelOp, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.eq(rhs)
}

/// Element-wise inequality comparison. `!=`
///
/// Examples:
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([1.2, 3.4, -0.5, -6.7]);
/// let b = dev.tensor([1.2, 0.0, 3.14, -6.7]);
/// let r = a.ne(&b);
/// assert_eq!(r.array(), [false, true, true, false]);
/// ```
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([-2, -1, 0, 1, 2]);
/// let r = a.ne(1);
/// assert_eq!(r.array(), [true, true, true, false, true]);
/// ```
pub fn ne<S: Shape, E: Unit, D: CmpKernel<NeKernelOp, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.ne(rhs)
}

/// Element-wise strictly greater than comparison. `>`
///
/// Examples:
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([1.2, 3.4, -0.5, -1.0]);
/// let b = dev.tensor([1.2, 0.0, 3.14, -6.7]);
/// let r = a.gt(&b);
/// assert_eq!(r.array(), [false, true, false, true]);
/// ```
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([-2, -1, 0, 1, 2]);
/// let r = a.gt(-1);
/// assert_eq!(r.array(), [false, false, true, true, true]);
/// ```
pub fn gt<S: Shape, E: Unit, D: CmpKernel<GtKernelOp, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.gt(rhs)
}

/// Element-wise greater than or equals comparison. `>=`
///
/// Examples:
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([1.2, 3.4, -0.5, -1.0]);
/// let b = dev.tensor([1.2, 0.0, 3.14, -6.7]);
/// let r = a.ge(&b);
/// assert_eq!(r.array(), [true, true, false, true]);
/// ```
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([-2, -1, 0, 1, 2]);
/// let r = a.ge(-1);
/// assert_eq!(r.array(), [false, true, true, true, true]);
/// ```
pub fn ge<S: Shape, E: Unit, D: CmpKernel<GeKernelOp, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.ge(rhs)
}

/// Element-wise strictly less than comparison. `<`
///
/// Examples:
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([1.2, 3.4, -0.5, -1.0]);
/// let b = dev.tensor([1.2, 0.0, 3.14, -6.7]);
/// let r = a.lt(&b);
/// assert_eq!(r.array(), [false, false, true, false]);
/// ```
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([-2, -1, 0, 1, 2]);
/// let r = a.lt(1);
/// assert_eq!(r.array(), [true, true, true, false, false]);
/// ```
pub fn lt<S: Shape, E: Unit, D: CmpKernel<LtKernelOp, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.lt(rhs)
}

/// Element-wise less than or equals comparison. `<=`
///
/// Examples:
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([1.2, 3.4, -0.5, -1.0]);
/// let b = dev.tensor([1.2, 0.0, 3.14, -6.7]);
/// let r = a.le(&b);
/// assert_eq!(r.array(), [true, false, true, false]);
/// ```
/// ```
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([-2, -1, 0, 1, 2]);
/// let r = a.le(1);
/// assert_eq!(r.array(), [true, true, true, true, false]);
/// ```
pub fn le<S: Shape, E: Unit, D: CmpKernel<LeKernelOp, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.le(rhs)
}

// Macro to reduce boilerplate of implementing comparison methods on Tensor.
macro_rules! impl_cmp_kernel_op {
    ($TraitName:tt, $FnName:tt, $TryFnName:tt, $KernelOp:tt, $doc:expr) => {
        pub trait $TraitName<Rhs>: HasErr {
            type Output;
            #[doc = $doc]
            fn $FnName(&self, rhs: Rhs) -> Self::Output {
                self.$TryFnName(rhs).unwrap()
            }
            #[doc = $doc]
            fn $TryFnName(&self, rhs: Rhs) -> Result<Self::Output, Self::Err>;
        }

        impl<S: Shape, E: Unit, D: CmpKernel<$KernelOp, E>, T: Tape<E, D>> $TraitName<&Self>
            for Tensor<S, E, D, T>
        {
            type Output = Tensor<S, bool, D, NoneTape>;
            #[doc = $doc]
            fn $TryFnName(&self, other: &Self) -> Result<Self::Output, D::Err> {
                try_cmp_op(self, other)
            }
        }

        impl<S: Shape, E: Unit, D: ScalarCmpKernel<$KernelOp, E>, T: Tape<E, D>> $TraitName<E>
            for Tensor<S, E, D, T>
        {
            type Output = Tensor<S, bool, D, NoneTape>;
            #[doc = $doc]
            fn $TryFnName(&self, other: E) -> Result<Self::Output, D::Err> {
                try_scalar_cmp_op(self, other)
            }
        }
    };
}

impl_cmp_kernel_op!(TryEq, eq, try_eq, EqKernelOp, "See [eq]");
impl_cmp_kernel_op!(TryNe, ne, try_ne, NeKernelOp, "See [ne]");
impl_cmp_kernel_op!(TryGt, gt, try_gt, GtKernelOp, "See [gt]");
impl_cmp_kernel_op!(TryGe, ge, try_ge, GeKernelOp, "See [ge]");
impl_cmp_kernel_op!(TryLt, lt, try_lt, LtKernelOp, "See [lt]");
impl_cmp_kernel_op!(TryLe, le, try_le, LeKernelOp, "See [le]");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::*, tensor::*, tests::*};

    type TestTensor<const R: usize, const C: usize, E> =
        Tensor<(Const<R>, Const<C>), E, TestDevice>;

    fn test_cmp<E: Unit, const R: usize, const C: usize, F>(
        a: [[E; C]; R],
        b: [[E; C]; R],
        cmp: F,
        expected: [[bool; C]; R],
    ) where
        F: Fn(&TestTensor<R, C, E>, &TestTensor<R, C, E>) -> [[bool; C]; R],
    {
        let dev: TestDevice = Default::default();
        let a = dev.tensor(a);
        let b = dev.tensor(b);
        let r = cmp(&a, &b);
        assert_eq!(r, expected);
    }

    fn test_scalar_cmp<E: Unit, const R: usize, const C: usize, F>(
        a: [[E; C]; R],
        cmp: F,
        expected: [[bool; C]; R],
    ) where
        F: Fn(&TestTensor<R, C, E>) -> [[bool; C]; R],
    {
        let dev: TestDevice = Default::default();
        let a = dev.tensor(a);
        assert_eq!(cmp(&a), expected);
    }

    #[test]
    fn test_eq() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]])
            .to_dtype::<TestDtype>();

        {
            let b = dev
                .tensor([[0.0, 2.0, -3.0], [4.0, 0.5, -0.0]])
                .to_dtype::<TestDtype>();
            let r = a.eq(&b);
            assert_eq!(r.array(), [[false, true, false], [true, false, true]]);
        }

        {
            let r = a.eq(0.0);
            assert_eq!(r.array(), [[false, false, true], [false, false, true]]);
        }
    }

    #[test]
    fn test_ne() {
        test_cmp::<TestDtype, 2, 3, _>(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            [[0.0, 2.0, -3.0], [4.0, 0.5, -0.0]],
            |a, b| a.ne(b).array(),
            [[true, false, true], [false, true, false]],
        );
    }

    // TODO Remove this attribute once Cuda supports integers
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_ne_not_dtype() {
        test_cmp(
            [[1, 2, 3], [0, 123, 5]],
            [[0, 2, -3], [-4, 123, 6]],
            |a, b| a.ne(b).array(),
            [[true, false, true], [true, false, true]],
        );
    }

    #[test]
    fn test_scalar_ne() {
        test_scalar_cmp::<TestDtype, 2, 2, _>(
            [[0.0, 1.2], [3.4, -5.6]],
            |a| a.ne(1.2).array(),
            [[true, false], [true, true]],
        );
    }

    #[test]
    fn test_gt() {
        test_cmp::<TestDtype, 2, 3, _>(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            [[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]],
            |a, b| a.gt(b).array(),
            [[true, false, false], [true, true, false]],
        );
    }

    // TODO Remove this attribute once Cuda supports integers
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_gt_not_dtype() {
        test_cmp(
            [[1, 2, 3], [0, 123, 5]],
            [[0, 2, -3], [-4, 123, 6]],
            |a, b| a.gt(b).array(),
            [[true, false, true], [true, false, false]],
        );
    }

    #[test]
    fn test_scalar_gt() {
        test_scalar_cmp::<TestDtype, 2, 2, _>(
            [[0.0, 1.2], [3.4, -5.6]],
            |a| a.gt(1.2).array(),
            [[false, false], [true, false]],
        );
    }

    #[test]
    fn test_ge() {
        test_cmp::<TestDtype, 2, 3, _>(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            [[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]],
            |a, b| a.ge(b).array(),
            [[true, true, false], [true, true, true]],
        );
    }

    // TODO Remove this attribute once Cuda supports integers
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_ge_not_dtype() {
        test_cmp(
            [[1, 2, 3], [0, 123, 5]],
            [[0, 2, -3], [-4, 123, 6]],
            |a, b| a.ge(b).array(),
            [[true, true, true], [true, true, false]],
        );
    }

    #[test]
    fn test_scalar_ge() {
        test_scalar_cmp::<TestDtype, 2, 2, _>(
            [[0.0, 1.2], [3.4, -5.6]],
            |a| a.ge(1.2).array(),
            [[false, true], [true, false]],
        );
    }

    #[test]
    fn test_lt() {
        test_cmp::<TestDtype, 2, 3, _>(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            [[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]],
            |a, b| a.lt(b).array(),
            [[false, false, true], [false, false, false]],
        );
    }

    // TODO Remove this attribute once Cuda supports integers
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_lt_not_dtype() {
        test_cmp(
            [[1, 2, 3], [0, 123, 5]],
            [[0, 2, -3], [-4, 123, 6]],
            |a, b| a.lt(b).array(),
            [[false, false, false], [false, false, true]],
        );
    }

    #[test]
    fn test_scalar_lt() {
        test_scalar_cmp::<TestDtype, 2, 2, _>(
            [[0.0, 1.2], [3.4, -5.6]],
            |a| a.lt(1.2).array(),
            [[true, false], [false, true]],
        );
    }

    #[test]
    fn test_le() {
        test_cmp::<TestDtype, 2, 3, _>(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            [[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]],
            |a, b| a.le(b).array(),
            [[false, true, true], [false, false, true]],
        );
    }

    // TODO Remove this attribute once Cuda supports integers
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_le_not_dtype() {
        test_cmp(
            [[1, 2, 3], [0, 123, 5]],
            [[0, 2, -3], [-4, 123, 6]],
            |a, b| a.le(b).array(),
            [[false, true, false], [false, true, true]],
        );
    }

    #[test]
    fn test_scalar_le() {
        test_scalar_cmp::<TestDtype, 2, 2, _>(
            [[0.0, 1.2], [3.4, -5.6]],
            |a| a.le(1.2).array(),
            [[true, true], [false, true]],
        );
    }

    #[test]
    #[should_panic]
    fn test_cmp_shape_mismatch() {
        let dev: TestDevice = Default::default();
        let a: Tensor<(usize, usize, usize), TestDtype, TestDevice> = dev.zeros_like(&(1, 2, 3));
        let b: Tensor<(usize, usize, usize), TestDtype, TestDevice> = dev.ones_like(&(2, 3, 4));
        a.eq(&b);
    }
}
