use super::concat_shape_along::TryConcatShapeAlong;
use crate::{shapes::*, tensor::*};

pub(crate) mod cpu_kernel;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_kernel;
#[cfg(feature = "webgpu")]
mod webgpu_kernel;

/// Concatenate two tensors along a given axis.
///
/// **Pytorch equivalent** `torch.concat`.
///
/// # [Const] dims **requires nightly**
///
/// Along Axis 0:
/// ```ignore
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
/// let b: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
/// let _: Tensor<Rank2<6, 4>, f32, _> = (a, b).concat_tensor_along(Axis::<0>);
/// ```
///
/// Along Axis 1:
/// ```ignore
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
/// let b: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
/// let _: Tensor<Rank2<3, 8>, f32, _> = (a, b).concat_tensor_along(Axis::<1>);
/// ```
///
/// # [usize] dims
/// Along Axis 0:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<(usize, Const<3>), f32, _> = dev.zeros_like(&(2, Const));
/// let b: Tensor<(usize, Const<3>), f32, _> = dev.zeros_like(&(4, Const));
/// let _: Tensor<Rank2<6, 3>, f32, _> = (a, b).concat_tensor_along(Axis::<0>).realize();
/// ```
///
/// Along Axis 1:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<(Const<2>, usize), f32, _> = dev.zeros_like(&(Const, 2));
/// let b: Tensor<(Const<2>, usize), f32, _> = dev.zeros_like(&(Const, 4));
/// let _: Tensor<Rank2<2, 6>, f32, _> = (a, b).concat_tensor_along(Axis::<1>).realize();
/// ```
pub trait TryConcatTensorAlong<Ax>: Sized {
    type Output;

    /// Concatenates self along the given axis.
    fn concat_tensor_along(self, ax: Ax) -> Self::Output {
        self.try_concat_tensor_along(ax).unwrap()
    }
    /// Fallibly concatenates self along the given axis.
    fn try_concat_tensor_along(self, ax: Ax) -> Result<Self::Output, Error>;
}

pub trait ConcatAlongKernel<E: Dtype>: Storage<E> {
    fn forward<A: Shape, B: Shape, C: Shape>(
        &self,
        ax: usize,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
        c: &mut Tensor<C, E, Self>,
    ) -> Result<(), Error>;

    fn backward<A: Shape, B: Shape>(
        &self,
        ax: usize,
        a: &GhostTensor<A, E, Self>,
        grad_a: &mut Self::Vec,
        b: &GhostTensor<B, E, Self>,
        grad_b: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

impl<A, B, Ax, E: Dtype, D, T: Tape<E, D>, R: Tape<E, D>> TryConcatTensorAlong<Ax>
    for (Tensor<A, E, D, T>, Tensor<B, E, D, R>)
where
    Ax: Axes<Array = [isize; 1]>,
    D: ConcatAlongKernel<E> + ZerosTensor<E>,
    A: Shape + HasAxes<Ax>,
    B: Shape<Concrete = A::Concrete> + HasAxes<Ax>,
    (A, B): TryConcatShapeAlong<Ax>,
    T: Merge<R>,
{
    type Output = Tensor<<(A, B) as TryConcatShapeAlong<Ax>>::Output, E, D, T>;

    fn try_concat_tensor_along(self, ax: Ax) -> Result<Self::Output, Error> {
        let (lhs, rhs) = self;

        let out_shape = (*lhs.shape(), *rhs.shape()).concat_shape_along(ax);
        let ax = Ax::as_array()[0] as usize;

        let (lhs, tape) = lhs.split_tape();
        let (rhs, rtape) = rhs.split_tape();
        let mut tape = tape.merge(rtape);

        let mut out = lhs.device.try_zeros_like(&out_shape)?;
        lhs.device.forward(ax, &lhs, &rhs, &mut out)?;

        let lhs_ghost = lhs.ghost();
        let rhs_ghost = rhs.ghost();
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (lhs_grad, rhs_grad, out_grad) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs.device
                .backward(ax, &lhs_ghost, lhs_grad, &rhs_ghost, rhs_grad, out_grad)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_concat_ax_0() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let b: Tensor<Rank3<3, 3, 4>, TestDtype, _> = dev.sample_normal();
        let a_dyn = a
            .leaky_trace()
            .try_realize::<(usize, Const<3>, Const<4>)>()
            .unwrap();
        let b_dyn = b
            .clone()
            .try_realize::<(usize, Const<3>, Const<4>)>()
            .unwrap();
        let c = (a_dyn, b_dyn).concat_tensor_along(Axis::<0>);
        let c = c.try_realize::<(Const<5>, Const<3>, Const<4>)>().unwrap();
        let a_arr = a.array();
        let b_arr = b.array();
        let c_arr = c.array();
        println!("{a_arr:?}");
        println!("{b_arr:?}");
        println!("{c_arr:?}");
        assert_eq!(c_arr[0], a_arr[0]);
        assert_eq!(c_arr[1], a_arr[1]);
        assert_eq!(c_arr[2], b_arr[0]);
        assert_eq!(c_arr[3], b_arr[1]);
        assert_eq!(c_arr[4], b_arr[2]);
        let concat_grads = c.exp().sum().backward();
        let a_grads = a.leaky_trace().exp().sum().backward();
        let b_grads = b.leaky_trace().exp().sum().backward();
        assert_close_to_tensor!(concat_grads.get(&a), a_grads.get(&a));
        assert_close_to_tensor!(concat_grads.get(&b), b_grads.get(&b));
    }

    #[test]
    fn test_concat_ax_1() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank3<2, 2, 4>, TestDtype, _> = dev.sample_normal();
        let b: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let a_dyn = a
            .leaky_trace()
            .try_realize::<(Const<2>, usize, Const<4>)>()
            .unwrap();
        let b_dyn = b
            .clone()
            .try_realize::<(Const<2>, usize, Const<4>)>()
            .unwrap();
        let c = (a_dyn, b_dyn).concat_tensor_along(Axis::<1>);
        let c = c.try_realize::<(Const<2>, Const<5>, Const<4>)>().unwrap();
        let a_arr = a.array();
        let b_arr = b.array();
        let c_arr = c.array();
        for i in 0..2 {
            assert_eq!(c_arr[i][0], a_arr[i][0]);
            assert_eq!(c_arr[i][1], a_arr[i][1]);
            assert_eq!(c_arr[i][2], b_arr[i][0]);
            assert_eq!(c_arr[i][3], b_arr[i][1]);
            assert_eq!(c_arr[i][4], b_arr[i][2]);
        }
        let concat_grads = c.exp().sum().backward();
        let a_grads = a.leaky_trace().exp().sum().backward();
        let b_grads = b.leaky_trace().exp().sum().backward();
        assert_close_to_tensor!(concat_grads.get(&a), a_grads.get(&a));
        assert_close_to_tensor!(concat_grads.get(&b), b_grads.get(&b));
    }

    #[test]
    fn test_concat_ax_2() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank3<2, 3, 2>, TestDtype, _> = dev.sample_normal();
        let b: Tensor<Rank3<2, 3, 3>, TestDtype, _> = dev.sample_normal();
        let a_dyn = a
            .leaky_trace()
            .try_realize::<(Const<2>, Const<3>, usize)>()
            .unwrap();
        let b_dyn = b
            .clone()
            .try_realize::<(Const<2>, Const<3>, usize)>()
            .unwrap();
        let c = (a_dyn, b_dyn).concat_tensor_along(Axis::<2>);
        let c = c.try_realize::<(Const<2>, Const<3>, Const<5>)>().unwrap();
        let a_arr = a.array();
        let b_arr = b.array();
        let c_arr = c.array();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(c_arr[i][j][0], a_arr[i][j][0]);
                assert_eq!(c_arr[i][j][1], a_arr[i][j][1]);
                assert_eq!(c_arr[i][j][2], b_arr[i][j][0]);
                assert_eq!(c_arr[i][j][3], b_arr[i][j][1]);
                assert_eq!(c_arr[i][j][4], b_arr[i][j][2]);
            }
        }
        let concat_grads = c.exp().sum().backward();
        let a_grads = a.leaky_trace().exp().sum().backward();
        let b_grads = b.leaky_trace().exp().sum().backward();
        assert_close_to_tensor!(concat_grads.get(&a), a_grads.get(&a));
        assert_close_to_tensor!(concat_grads.get(&b), b_grads.get(&b));
    }
}
