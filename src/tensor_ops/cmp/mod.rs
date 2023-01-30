use crate::{
    gradients::{NoneTape, Tape},
    shapes::{Shape, Unit},
    tensor::{DeviceStorage, Tensor},
};

mod cpu_kernels;
#[cfg(feature = "cuda")]
mod cuda_kernels;

pub trait CmpKernel<Op, E: Unit>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, bool>, Self::Err>;
}

fn try_cmp_op<Op, S: Shape, E: Unit, D: CmpKernel<Op, E>, T: Tape<D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
    let storage = lhs.device.forward(&lhs.storage, &rhs.storage)?;
    let out = lhs.device.upgrade(storage);
    Ok(out)
}

pub trait ScalarCmpKernel<Op, E: Unit>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        tensor: &Self::Storage<S, E>,
        scalar: E,
    ) -> Result<Self::Storage<S, bool>, Self::Err>;
}

fn try_scalar_cmp_op<Op, S: Shape, E: Unit, D: ScalarCmpKernel<Op, E>, T: Tape<D>>(
    tensor: &Tensor<S, E, D, T>,
    scalar: E,
) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
    let storage = tensor.device.forward(&tensor.storage, scalar)?;
    let out = tensor.device.upgrade(storage);
    Ok(out)
}

pub enum EqKernelOp {}
pub enum NeKernelOp {}
pub enum GtKernelOp {}
pub enum GeKernelOp {}
pub enum LtKernelOp {}
pub enum LeKernelOp {}

///
pub fn eq<S: Shape, E: Unit, D: CmpKernel<EqKernelOp, E>, T: Tape<D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.eq(rhs)
}

impl<S: Shape, E: Unit, D: CmpKernel<EqKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_eq(&self, other: &Self) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_cmp_op(self, other)
    }

    pub fn eq(&self, other: &Self) -> Tensor<S, bool, D, NoneTape> {
        self.try_eq(other).unwrap()
    }
}

impl<S: Shape, E: Unit, D: ScalarCmpKernel<EqKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_scalar_eq(&self, scalar: E) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_scalar_cmp_op(self, scalar)
    }

    pub fn scalar_eq(&self, scalar: E) -> Tensor<S, bool, D, NoneTape> {
        self.try_scalar_eq(scalar).unwrap()
    }
}

///
pub fn ne<S: Shape, E: Unit, D: CmpKernel<NeKernelOp, E>, T: Tape<D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.ne(rhs)
}

impl<S: Shape, E: Unit, D: CmpKernel<NeKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_ne(&self, other: &Self) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_cmp_op(self, other)
    }

    pub fn ne(&self, other: &Self) -> Tensor<S, bool, D, NoneTape> {
        self.try_ne(other).unwrap()
    }
}

impl<S: Shape, E: Unit, D: ScalarCmpKernel<NeKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_scalar_ne(&self, scalar: E) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_scalar_cmp_op(self, scalar)
    }

    pub fn scalar_ne(&self, scalar: E) -> Tensor<S, bool, D, NoneTape> {
        self.try_scalar_ne(scalar).unwrap()
    }
}

///
pub fn gt<S: Shape, E: Unit, D: CmpKernel<GtKernelOp, E>, T: Tape<D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.gt(rhs)
}

impl<S: Shape, E: Unit, D: CmpKernel<GtKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_gt(&self, other: &Self) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_cmp_op(self, other)
    }

    pub fn gt(&self, other: &Self) -> Tensor<S, bool, D, NoneTape> {
        self.try_gt(other).unwrap()
    }
}

impl<S: Shape, E: Unit, D: ScalarCmpKernel<GtKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_scalar_gt(&self, scalar: E) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_scalar_cmp_op(self, scalar)
    }

    pub fn scalar_gt(&self, scalar: E) -> Tensor<S, bool, D, NoneTape> {
        self.try_scalar_gt(scalar).unwrap()
    }
}

///
pub fn ge<S: Shape, E: Unit, D: CmpKernel<GeKernelOp, E>, T: Tape<D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.ge(rhs)
}

impl<S: Shape, E: Unit, D: CmpKernel<GeKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_ge(&self, other: &Self) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_cmp_op(self, other)
    }

    pub fn ge(&self, other: &Self) -> Tensor<S, bool, D, NoneTape> {
        self.try_ge(other).unwrap()
    }
}

impl<S: Shape, E: Unit, D: ScalarCmpKernel<GeKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_scalar_ge(&self, scalar: E) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_scalar_cmp_op(self, scalar)
    }

    pub fn scalar_ge(&self, scalar: E) -> Tensor<S, bool, D, NoneTape> {
        self.try_scalar_ge(scalar).unwrap()
    }
}

///
pub fn lt<S: Shape, E: Unit, D: CmpKernel<LtKernelOp, E>, T: Tape<D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.lt(rhs)
}

impl<S: Shape, E: Unit, D: CmpKernel<LtKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_lt(&self, other: &Self) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_cmp_op(self, other)
    }

    pub fn lt(&self, other: &Self) -> Tensor<S, bool, D, NoneTape> {
        self.try_lt(other).unwrap()
    }
}

impl<S: Shape, E: Unit, D: ScalarCmpKernel<LtKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_scalar_lt(&self, scalar: E) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_scalar_cmp_op(self, scalar)
    }

    pub fn scalar_lt(&self, scalar: E) -> Tensor<S, bool, D, NoneTape> {
        self.try_scalar_lt(scalar).unwrap()
    }
}

///
pub fn le<S: Shape, E: Unit, D: CmpKernel<LeKernelOp, E>, T: Tape<D>>(
    lhs: &Tensor<S, E, D, T>,
    rhs: &Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.le(rhs)
}

impl<S: Shape, E: Unit, D: CmpKernel<LeKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_le(&self, other: &Self) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_cmp_op(self, other)
    }

    pub fn le(&self, other: &Self) -> Tensor<S, bool, D, NoneTape> {
        self.try_le(other).unwrap()
    }
}

impl<S: Shape, E: Unit, D: ScalarCmpKernel<LeKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_scalar_le(&self, scalar: E) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_scalar_cmp_op(self, scalar)
    }

    pub fn scalar_le(&self, scalar: E) -> Tensor<S, bool, D, NoneTape> {
        self.try_scalar_le(scalar).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::TestDevice};

    #[test]
    fn test_eq() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]);
        let b = dev.tensor([[0.0, 2.0, -3.0], [4.0, 0.5, -0.0]]);

        let r = a.eq(&b);
        assert_eq!(r.array(), [[false, true, false], [true, false, true]]);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_eq_not_dtype() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
        let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);

        let r = a.eq(&b);
        assert_eq!(r.array(), [[false, true, false], [false, true, false]]);
    }

    #[test]
    fn test_scalar_eq() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.0, 1.2], [3.4, -5.6]]);
        let r = a.scalar_eq(1.2);
        assert_eq!(r.array(), [[false, true], [false, false]]);
    }

    #[test]
    fn test_ne() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]);
        let b = dev.tensor([[0.0, 2.0, -3.0], [4.0, 0.5, -0.0]]);

        let r = a.ne(&b);
        assert_eq!(r.array(), [[true, false, true], [false, true, false]]);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_ne_not_dtype() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
        let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);

        let r = a.ne(&b);
        assert_eq!(r.array(), [[true, false, true], [true, false, true]]);
    }

    #[test]
    fn test_scalar_ne() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.0, 1.2], [3.4, -5.6]]);
        let r = a.scalar_ne(1.2);
        assert_eq!(r.array(), [[true, false], [true, true]]);
    }

    #[test]
    fn test_gt() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]);
        let b = dev.tensor([[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]]);

        let r = a.gt(&b);
        assert_eq!(r.array(), [[true, false, false], [true, true, false]]);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_gt_not_dtype() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
        let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);

        let r = a.gt(&b);
        assert_eq!(r.array(), [[true, false, true], [true, false, false]]);
    }

    #[test]
    fn test_scalar_gt() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.0, 1.2], [3.4, -5.6]]);
        let r = a.scalar_gt(1.2);
        assert_eq!(r.array(), [[false, false], [true, false]]);
    }

    #[test]
    fn test_ge() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]);
        let b = dev.tensor([[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]]);

        let r = a.ge(&b);
        assert_eq!(r.array(), [[true, true, false], [true, true, true]]);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_ge_not_dtype() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
        let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);

        let r = a.ge(&b);
        assert_eq!(r.array(), [[true, true, true], [true, true, false]]);
    }

    #[test]
    fn test_scalar_ge() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.0, 1.2], [3.4, -5.6]]);
        let r = a.scalar_ge(1.2);
        assert_eq!(r.array(), [[false, true], [true, false]]);
    }

    #[test]
    fn test_lt() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]);
        let b = dev.tensor([[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]]);

        let r = a.lt(&b);
        assert_eq!(r.array(), [[false, false, true], [false, false, false]]);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_lt_not_dtype() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
        let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);

        let r = a.lt(&b);
        assert_eq!(r.array(), [[false, false, false], [false, false, true]]);
    }

    #[test]
    fn test_scalar_lt() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.0, 1.2], [3.4, -5.6]]);
        let r = a.scalar_lt(1.2);
        assert_eq!(r.array(), [[true, false], [false, true]]);
    }

    #[test]
    fn test_le() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]);
        let b = dev.tensor([[0.0, 2.0, 3.1], [-4.0, -5.5, -0.0]]);

        let r = a.le(&b);
        assert_eq!(r.array(), [[false, true, true], [false, false, true]]);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_le_not_dtype() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
        let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);

        let r = a.le(&b);
        assert_eq!(r.array(), [[false, true, false], [false, true, true]]);
    }

    #[test]
    fn test_scalar_le() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.0, 1.2], [3.4, -5.6]]);
        let r = a.scalar_le(1.2);
        assert_eq!(r.array(), [[true, true], [false, true]]);
    }
}
