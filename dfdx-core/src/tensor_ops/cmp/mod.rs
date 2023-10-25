use crate::{
    shapes::{HasShape, Shape},
    tensor::{HasErr, NoneTape, Storage, Tape, Tensor},
};

mod cpu_kernels;
#[cfg(feature = "cuda")]
mod cuda_kernels;

pub trait CmpKernel<Op, E>: Storage<E> + Storage<bool> {
    fn forward<S: Shape, T>(
        &self,
        lhs: &Tensor<S, E, Self, T>,
        rhs: &Tensor<S, E, Self, T>,
    ) -> Result<Tensor<S, bool, Self>, Self::Err>;
}

fn try_cmp_op<Op, S: Shape, E, D: CmpKernel<Op, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
    assert_eq!(lhs.shape(), rhs.shape());
    lhs.device.forward(lhs, rhs)
}

pub trait ScalarCmpKernel<Op, E>: Storage<E> + Storage<bool> {
    fn forward<S: Shape, T>(
        &self,
        tensor: &Tensor<S, E, Self, T>,
        scalar: E,
    ) -> Result<Tensor<S, bool, Self>, Self::Err>;
}

fn try_scalar_cmp_op<Op, S: Shape, E, D: ScalarCmpKernel<Op, E>, T: Tape<E, D>>(
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
pub fn eq<S: Shape, E, D: CmpKernel<EqKernelOp, E>, T: Tape<E, D>>(
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
pub fn ne<S: Shape, E, D: CmpKernel<NeKernelOp, E>, T: Tape<E, D>>(
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
pub fn gt<S: Shape, E, D: CmpKernel<GtKernelOp, E>, T: Tape<E, D>>(
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
pub fn ge<S: Shape, E, D: CmpKernel<GeKernelOp, E>, T: Tape<E, D>>(
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
pub fn lt<S: Shape, E, D: CmpKernel<LtKernelOp, E>, T: Tape<E, D>>(
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
pub fn le<S: Shape, E, D: CmpKernel<LeKernelOp, E>, T: Tape<E, D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.le(rhs)
}

// Macro to reduce boilerplate of implementing comparison methods on Tensor.
macro_rules! impl_cmp_kernel_op {
    ($TraitName:tt, $FnName:tt, $TryFnName:tt, $KernelOp:tt, $doc:expr, $ScalarFnName:tt, $TryScalarFnName:tt) => {
        pub trait $TraitName<Rhs>: HasErr {
            type Output;
            #[doc = $doc]
            fn $FnName(&self, rhs: Rhs) -> Self::Output {
                self.$TryFnName(rhs).unwrap()
            }
            #[doc = $doc]
            fn $TryFnName(&self, rhs: Rhs) -> Result<Self::Output, Self::Err>;
        }

        impl<S: Shape, E, D: CmpKernel<$KernelOp, E>, T: Tape<E, D>> $TraitName<&Self>
            for Tensor<S, E, D, T>
        {
            type Output = Tensor<S, bool, D, NoneTape>;
            #[doc = $doc]
            fn $TryFnName(&self, other: &Self) -> Result<Self::Output, D::Err> {
                try_cmp_op(self, other)
            }
        }

        impl<S: Shape, E, D: ScalarCmpKernel<$KernelOp, E>, T: Tape<E, D>> $TraitName<E>
            for Tensor<S, E, D, T>
        {
            type Output = Tensor<S, bool, D, NoneTape>;
            #[doc = $doc]
            fn $TryFnName(&self, other: E) -> Result<Self::Output, D::Err> {
                try_scalar_cmp_op(self, other)
            }
        }

        #[cfg(feature = "f16")]
        impl<S: Shape, D: ScalarCmpKernel<$KernelOp, half::f16>, T: Tape<half::f16, D>>
            $TraitName<f32> for Tensor<S, half::f16, D, T>
        {
            type Output = Tensor<S, bool, D, NoneTape>;
            #[doc = $doc]
            fn $TryFnName(&self, other: f32) -> Result<Self::Output, D::Err> {
                try_scalar_cmp_op(self, half::f16::from_f32(other))
            }
        }

        #[cfg(feature = "f16")]
        impl<
                S: Shape,
                D: ScalarCmpKernel<$KernelOp, crate::dtypes::AMP<half::f16>>,
                T: Tape<crate::dtypes::AMP<half::f16>, D>,
            > $TraitName<f32> for Tensor<S, crate::dtypes::AMP<half::f16>, D, T>
        {
            type Output = Tensor<S, bool, D, NoneTape>;
            #[doc = $doc]
            fn $TryFnName(&self, other: f32) -> Result<Self::Output, D::Err> {
                try_scalar_cmp_op(self, crate::dtypes::AMP(half::f16::from_f32(other)))
            }
        }

        impl<S: Shape, E, D: ScalarCmpKernel<$KernelOp, E>, T: Tape<E, D>> Tensor<S, E, D, T> {
            #[doc = $doc]
            #[deprecated = "You can now use the non-scalar method for both tensors & scalars."]
            pub fn $ScalarFnName(&self, other: E) -> Tensor<S, bool, D, NoneTape> {
                try_scalar_cmp_op(self, other).unwrap()
            }

            #[doc = $doc]
            #[deprecated = "You can now use the non-scalar method for both tensors & scalars."]
            pub fn $TryScalarFnName(
                &self,
                other: E,
            ) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
                try_scalar_cmp_op(self, other)
            }
        }
    };
}

impl_cmp_kernel_op!(
    TryEq,
    eq,
    try_eq,
    EqKernelOp,
    "See [eq]",
    scalar_eq,
    try_scalar_eq
);
impl_cmp_kernel_op!(
    TryNe,
    ne,
    try_ne,
    NeKernelOp,
    "See [ne]",
    scalar_ne,
    try_scalar_ne
);
impl_cmp_kernel_op!(
    TryGt,
    gt,
    try_gt,
    GtKernelOp,
    "See [gt]",
    scalar_gt,
    try_scalar_gt
);
impl_cmp_kernel_op!(
    TryGe,
    ge,
    try_ge,
    GeKernelOp,
    "See [ge]",
    scalar_ge,
    try_scalar_ge
);
impl_cmp_kernel_op!(
    TryLt,
    lt,
    try_lt,
    LtKernelOp,
    "See [lt]",
    scalar_lt,
    try_scalar_lt
);
impl_cmp_kernel_op!(
    TryLe,
    le,
    try_le,
    LeKernelOp,
    "See [le]",
    scalar_le,
    try_scalar_le
);

#[cfg(test)]
mod tests {
    use super::*;
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
        let r = a.eq(1.2);
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
        let r = a.ne(1.2);
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
        let r = a.gt(1.2);
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
        let r = a.ge(1.2);
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
        let r = a.lt(1.2);
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
        let r = a.le(1.2);
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
