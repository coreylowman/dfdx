use crate::{
    shapes::{Axes, Dtype, HasAxes, HasShape, Shape},
    tensor::*,
};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RollOp {
    axis: usize,
    amount: usize,
}

pub trait RollKernel<E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        op: RollOp,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;
    fn backward<S: Shape>(
        &self,
        op: RollOp,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

/// Shifts data along an axis by a specified amount.
///
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([1.0, 2.0, 3.0, 4.0]);
/// let r = t.roll::<Axis<0>>(1);
/// assert_eq!(r.array(), [4.0, 1.0, 2.0, 3.0]);
/// ```
///
/// Won't compile if you try to roll an axis that doesn't exist:
/// ```compile_fail
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([1.0, 2.0, 3.0, 4.0]);
/// let r = t.roll::<Axis<3>>(1);
/// assert_eq!(r.array(), [4.0, 1.0, 2.0, 3.0]);
/// ```
pub trait Roll: HasShape + HasErr {
    /// Shifts data along an axis by a specified amount.
    fn roll<Ax: Axes<Array = [isize; 1]>>(self, amount: usize) -> Self
    where
        Self::Shape: HasAxes<Ax>,
    {
        self.try_roll::<Ax>(amount).unwrap()
    }

    /// Shifts data along an axis by a specified amount.
    fn try_roll<Ax: Axes<Array = [isize; 1]>>(self, amount: usize) -> Result<Self, Self::Err>
    where
        Self::Shape: HasAxes<Ax>;
}

impl<S: Shape, E: Dtype, D: RollKernel<E>, T: Tape<E, D>> Roll for Tensor<S, E, D, T> {
    fn try_roll<Ax: Axes<Array = [isize; 1]>>(self, amount: usize) -> Result<Self, D::Err>
    where
        S: HasAxes<Ax>,
    {
        let op = RollOp {
            axis: Ax::as_array()[0] as usize,
            amount,
        };
        let (t, mut tape) = self.split_tape();
        let out = t.device.forward(op, &t)?;
        let inp_ghost = t.ghost();
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            t.device.backward(op, &t, grad_inp, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::*, tensor_ops::*, tests::*};

    #[test]
    fn test_roll_3d_axis_2() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<5>, TestDtype, _> = dev.tensor([-0.3, -0.15, 0.0, 0.15, 0.2]);
        let y = t
            .leaky_trace()
            .broadcast::<Rank3<2, 3, 5>, _>()
            .roll::<Axis<2>>(2);
        assert_close_to_literal!(y, [[[0.15, 0.2, -0.3, -0.15, 0.0]; 3]; 2]);
        let grads = y.exp().mean().backward();
        assert_close_to_literal!(
            grads.get(&t),
            [0.14816365, 0.1721416, 0.2, 0.23236685, 0.24428058]
        );
    }

    #[test]
    fn test_roll_3d_first_two_axes() {
        let dev: TestDevice = Default::default();
        let t: Tensor<Rank1<5>, TestDtype, _> = dev.tensor([1.0, 2.0, 3.0, 4.0, 5.0]);
        let y0 = t
            .leaky_trace()
            .broadcast::<Rank3<2, 3, 5>, _>()
            .roll::<Axis<0>>(3);
        assert_close_to_literal!(y0, [[[1.0, 2.0, 3.0, 4.0, 5.0]; 3]; 2]);
        let y1 = t
            .leaky_trace()
            .broadcast::<Rank3<2, 3, 5>, _>()
            .roll::<Axis<1>>(3);
        assert_close_to_literal!(y1, [[[1.0, 2.0, 3.0, 4.0, 5.0]; 3]; 2]);

        let g0 = y0.exp().mean().backward();
        let g1 = y1.exp().mean().backward();
        assert_eq!(g0.get(&t).array(), g1.get(&t).array());
    }
}
