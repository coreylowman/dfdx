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

pub(crate) fn try_cmp_op<Op, S: Shape, E: Unit, D: CmpKernel<Op, E>, T: Tape<D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, T>,
) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
    let storage = lhs.device.forward(&lhs.storage, &rhs.storage)?;
    let out = lhs.device.upgrade(storage);
    Ok(out)
}

pub enum EqKernelOp {}

impl<S: Shape, E: Unit, D: CmpKernel<EqKernelOp, E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn try_eq(self, other: Self) -> Result<Tensor<S, bool, D, NoneTape>, D::Err> {
        try_cmp_op(self, other)
    }

    pub fn eq(self, other: Self) -> Tensor<S, bool, D, NoneTape> {
        self.try_eq(other).unwrap()
    }
}

pub fn eq<S: Shape, E: Unit, D: CmpKernel<EqKernelOp, E>, T: Tape<D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, T>,
) -> Tensor<S, bool, D, NoneTape> {
    lhs.eq(rhs)
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::TestDevice};

    #[test]
    fn test_eq() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]);
        let b = dev.tensor([[0.0, 2.0, -3.0], [4.0, 0.5, -0.0]]);

        let result = a.eq(b);
        assert_eq!(result.array(), [[false, true, false], [true, false, true]]);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_eq_not_dtype() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[1, 2, 3], [0, 123, 5]]);
        let b = dev.tensor([[0, 2, -3], [-4, 123, 6]]);

        let result = a.eq(b);
        assert_eq!(result.array(), [[false, true, false], [false, true, false]]);
    }
}
