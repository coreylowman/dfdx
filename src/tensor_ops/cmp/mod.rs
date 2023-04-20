use crate::{
    shapes::{HasShape, Shape, Unit},
    tensor::{DeviceStorage, NoneTape, Tape, Tensor},
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
/// let r = a.scalar_eq(1);
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
/// let r = a.scalar_ne(1);
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
/// let r = a.scalar_gt(-1);
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
/// let r = a.scalar_ge(-1);
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
/// let r = a.scalar_lt(1);
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
/// let r = a.scalar_le(1);
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
    ($kernel_op:ty, $try_op:ident, $op:ident, $try_scalar_op:ident, $scalar_op:ident, $doc:expr) => {
        impl<S: Shape, E: Unit, D: CmpKernel<$kernel_op, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
            #[doc = $doc]
            pub fn $try_op(&self, other: &Self) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
                try_cmp_op(self, other)
            }

            #[doc = $doc]
            pub fn $op(&self, other: &Self) -> Tensor<S, bool, D, NoneTape> {
                self.$try_op(other).unwrap()
            }
        }

        impl<S: Shape, E: Unit, D: ScalarCmpKernel<$kernel_op, E>, T: Tape<E, D>>
            Tensor<S, E, D, T>
        {
            #[doc = $doc]
            pub fn $try_scalar_op(
                &self,
                scalar: impl Into<E>,
            ) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
                try_scalar_cmp_op(self, scalar.into())
            }

            #[doc = $doc]
            pub fn $scalar_op(&self, scalar: impl Into<E>) -> Tensor<S, bool, D, NoneTape> {
                self.$try_scalar_op(scalar).unwrap()
            }
        }
    };
}

impl_cmp_kernel_op!(EqKernelOp, try_eq, eq, try_scalar_eq, scalar_eq, "See [eq]");
impl_cmp_kernel_op!(NeKernelOp, try_ne, ne, try_scalar_ne, scalar_ne, "See [ne]");
impl_cmp_kernel_op!(GtKernelOp, try_gt, gt, try_scalar_gt, scalar_gt, "See [gt]");
impl_cmp_kernel_op!(GeKernelOp, try_ge, ge, try_scalar_ge, scalar_ge, "See [ge]");
impl_cmp_kernel_op!(LtKernelOp, try_lt, lt, try_scalar_lt, scalar_lt, "See [lt]");
impl_cmp_kernel_op!(LeKernelOp, try_le, le, try_scalar_le, scalar_le, "See [le]");

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::*};

    #[test]
    fn test_eq() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.0, 2.0, -3.0], [4.0, 0.5, -0.0]])
            .to_dtype::<TestDtype>();
        let r = a.eq(&b);
        assert_eq!(r.array(), [[false, true, false], [true, false, true]]);

        #[cfg(not(feature = "cuda"))]
        {
            let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
            let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);
            let r = a.eq(&b);
            assert_eq!(r.array(), [[false, true, false], [false, true, false]]);
        }
    }

    #[test]
    fn test_scalar_eq() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.0, 1.2], [3.4, -5.6]])
            .to_dtype::<TestDtype>();
        let r = a.scalar_eq(1.2);
        assert_eq!(r.array(), [[false, true], [false, false]]);
    }

    #[test]
    fn test_ne() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.0, 2.0, -3.0], [4.0, 0.5, -0.0]])
            .to_dtype::<TestDtype>();
        let r = a.ne(&b);
        assert_eq!(r.array(), [[true, false, true], [false, true, false]]);

        #[cfg(not(feature = "cuda"))]
        {
            let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
            let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);
            let r = a.ne(&b);
            assert_eq!(r.array(), [[true, false, true], [true, false, true]]);
        }
    }

    #[test]
    fn test_scalar_ne() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.0, 1.2], [3.4, -5.6]])
            .to_dtype::<TestDtype>();
        let r = a.scalar_ne(1.2);
        assert_eq!(r.array(), [[true, false], [true, true]]);
    }

    #[test]
    fn test_gt() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]])
            .to_dtype::<TestDtype>();
        let r = a.gt(&b);
        assert_eq!(r.array(), [[true, false, false], [true, true, false]]);

        #[cfg(not(feature = "cuda"))]
        {
            let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
            let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);
            let r = a.gt(&b);
            assert_eq!(r.array(), [[true, false, true], [true, false, false]]);
        }
    }

    #[test]
    fn test_scalar_gt() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.0, 1.2], [3.4, -5.6]])
            .to_dtype::<TestDtype>();
        let r = a.scalar_gt(1.2);
        assert_eq!(r.array(), [[false, false], [true, false]]);
    }

    #[test]
    fn test_ge() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]])
            .to_dtype::<TestDtype>();
        let r = a.ge(&b);
        assert_eq!(r.array(), [[true, true, false], [true, true, true]]);

        #[cfg(not(feature = "cuda"))]
        {
            let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
            let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);
            let r = a.ge(&b);
            assert_eq!(r.array(), [[true, true, true], [true, true, false]]);
        }
    }

    #[test]
    fn test_scalar_ge() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.0, 1.2], [3.4, -5.6]])
            .to_dtype::<TestDtype>();
        let r = a.scalar_ge(1.2);
        assert_eq!(r.array(), [[false, true], [true, false]]);
    }

    #[test]
    fn test_lt() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]])
            .to_dtype::<TestDtype>();
        let r = a.lt(&b);
        assert_eq!(r.array(), [[false, false, true], [false, false, false]]);

        #[cfg(not(feature = "cuda"))]
        {
            let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
            let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);
            let r = a.lt(&b);
            assert_eq!(r.array(), [[false, false, false], [false, false, true]]);
        }
    }

    #[test]
    fn test_scalar_lt() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.0, 1.2], [3.4, -5.6]])
            .to_dtype::<TestDtype>();
        let r = a.scalar_lt(1.2);
        assert_eq!(r.array(), [[true, false], [false, true]]);
    }

    #[test]
    fn test_le() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
            .to_dtype::<TestDtype>();
        let b = dev
            .tensor([[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]])
            .to_dtype::<TestDtype>();
        let r = a.le(&b);
        assert_eq!(r.array(), [[false, true, true], [false, false, true]]);

        #[cfg(not(feature = "cuda"))]
        {
            let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
            let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);
            let r = a.le(&b);
            assert_eq!(r.array(), [[false, true, false], [false, true, true]]);
        }
    }

    #[test]
    fn test_scalar_le() {
        let dev: TestDevice = Default::default();
        let a = dev
            .tensor([[0.0, 1.2], [3.4, -5.6]])
            .to_dtype::<TestDtype>();
        let r = a.scalar_le(1.2);
        assert_eq!(r.array(), [[true, true], [false, true]]);
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
