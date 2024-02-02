use super::split_shape_along::TrySplitShapeAlong;
use crate::{shapes::*, tensor::*};

pub(crate) mod cpu_kernel;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_kernel;
#[cfg(feature = "webgpu")]
mod webgpu_kernel;

/// Split a tensor in two along a given axis.
///
/// This is the reverse of [TryConcatTensorAlong::concat_tensor_along].
///
/// # [Const] dims **requires nightly**
///
/// Along Axis 0:
/// ```ignore
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let ab: Tensor<Rank2<5, 4>, f32, _> = dev.zeros();
/// let (a, b, _tape): (
///     Tensor<Rank2<2, 4>, f32, _>,
///     Tensor<Rank2<3, 4>, f32, _>,
///     _
/// ) = ab.split_tensor_along(Axis::<0>, Const::<2>, Const::<3>);
/// ```
///
/// Along Axis 1:
/// ```ignore
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let ab: Tensor<Rank2<4, 5>, f32, _> = dev.zeros();
/// let (a, b, _tape): (
///     Tensor<Rank2<4, 2>, f32, _>,
///     Tensor<Rank2<4, 3>, f32, _>,
///     _
/// ) = ab.split_tensor_along(Axis::<1>, Const::<2>, Const::<3>);
/// ```
///
/// # [usize] dims
/// Along Axis 0:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let ab: Tensor<(usize, Const::<4>), f32, _> = dev.zeros_like(&(5, Const));
/// let (a, b, _tape): (
///     Tensor<(usize, Const::<4>), f32, _>,
///     Tensor<(usize, Const::<4>), f32, _>,
///     _
/// ) = ab.split_tensor_along(Axis::<0>, 2, 3);
/// let a: Tensor<Rank2<2, 4>, f32, _> = a.realize();
/// let b: Tensor<Rank2<3, 4>, f32, _> = b.realize();
/// ```
///
/// Along Axis 1:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let ab: Tensor<(Const::<4>, usize), f32, _> = dev.zeros_like(&(Const, 5));
/// let (a, b, _tape): (
///     Tensor<(Const::<4>, usize), f32, _>,
///     Tensor<(Const::<4>, usize), f32, _>,
///     _
/// ) = ab.split_tensor_along(Axis::<1>, 2, 3);
/// let a: Tensor<Rank2<4, 2>, f32, _> = a.realize();
/// let b: Tensor<Rank2<4, 3>, f32, _> = b.realize();
/// ```
pub trait TrySplitTensorAlong<Ax, A: Dim, B: Dim>: Sized {
    type Output;

    /// Splits self along the given axis.
    fn split_tensor_along(self, ax: Ax, a: A, b: B) -> Self::Output {
        self.try_split_tensor_along(ax, a, b).unwrap()
    }
    /// Fallibly splits self along the given axis.
    fn try_split_tensor_along(self, ax: Ax, a: A, b: B) -> Result<Self::Output, Error>;
}

#[derive(Debug, Clone)]
pub enum AorB {
    A,
    B,
}

pub trait SplitAlongKernel<E: Dtype>: Storage<E> {
    fn forward<AB: Shape, A: Shape, B: Shape>(
        &self,
        ax: usize,
        ab: &Tensor<AB, E, Self>,
        a: &mut Tensor<A, E, Self>,
        b: &mut Tensor<B, E, Self>,
    ) -> Result<(), Error>;

    #[allow(clippy::too_many_arguments)]
    fn backward<AB: Shape, A: Shape, B: Shape>(
        &self,
        ax: usize,
        ab: &GhostTensor<AB, E, Self>,
        grad_ab: &mut Self::Vec,
        a: &GhostTensor<A, E, Self>,
        b: &GhostTensor<B, E, Self>,
        a_or_b: AorB,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

impl<A, B, AS, BS, AB, Ax, E: Dtype, D, T: Tape<E, D>> TrySplitTensorAlong<Ax, A, B>
    for Tensor<AB, E, D, T>
where
    Ax: Axes<Array = [isize; 1]>,
    A: Dim,
    B: Dim,
    AS: Shape,
    BS: Shape,
    AB: Shape + TrySplitShapeAlong<Ax, A, B, Output = (AS, BS)>,
    D: SplitAlongKernel<E> + ZerosTensor<E>,
{
    type Output = (Tensor<AS, E, D, T>, Tensor<BS, E, D, T>, T);

    fn try_split_tensor_along(self, ax: Ax, a: A, b: B) -> Result<Self::Output, Error> {
        let device = self.device.clone();
        let (a_shape, b_shape) = (*self.shape()).try_split_shape_along(ax, a, b)?;
        let ax = Ax::as_array()[0] as usize;

        let (ab, tape) = self.split_tape();

        let mut at: Tensor<AS, E, D, NoneTape> = device.try_zeros_like(&a_shape)?;
        let mut bt: Tensor<BS, E, D, NoneTape> = device.try_zeros_like(&b_shape)?;

        ab.device.forward(ax, &ab, &mut at, &mut bt)?;

        let mut ta = T::default();
        let mut tb = T::default();

        let device_b = device.clone();

        let ab_ghost = ab.ghost();
        let a_ghost = at.ghost();
        let b_ghost = bt.ghost();
        ta.add_backward_op(move |grads| {
            grads.try_alloc_for(&ab_ghost)?;
            grads.try_alloc_for(&a_ghost)?;
            let (ab_grad, a_grad) = grads.mut_and_ref(&ab_ghost, &a_ghost);
            device.backward(ax, &ab_ghost, ab_grad, &a_ghost, &b_ghost, AorB::A, a_grad)
        });

        let ab_ghost = ab.ghost();
        let a_ghost = at.ghost();
        let b_ghost = bt.ghost();
        tb.add_backward_op(move |grads| {
            grads.try_alloc_for(&ab_ghost)?;
            grads.try_alloc_for(&b_ghost)?;
            let (ab_grad, b_grad) = grads.mut_and_ref(&ab_ghost, &b_ghost);
            device_b.backward(ax, &ab_ghost, ab_grad, &a_ghost, &b_ghost, AorB::B, b_grad)
        });

        let at = at.put_tape(ta);
        let bt = bt.put_tape(tb);
        Ok((at, bt, tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_split_ax_0() {
        let dev: TestDevice = Default::default();
        let ab: Tensor<Rank3<5, 3, 4>, TestDtype, _> = dev.sample_normal();
        let ab_dyn = ab
            .leaky_trace()
            .try_realize::<(usize, Const<3>, Const<4>)>()
            .unwrap();
        let (a, b, _tape) = ab_dyn.split_tensor_along(Axis::<0>, 2, 3);
        let a = a.try_realize::<(Const<2>, Const<3>, Const<4>)>().unwrap();
        let b = b.try_realize::<(Const<3>, Const<3>, Const<4>)>().unwrap();
        let ab_arr = ab.array();
        let a_arr = a.array();
        let b_arr = b.array();
        println!("{a_arr:?}");
        println!("{b_arr:?}");
        println!("{ab_arr:?}");

        assert_eq!(ab_arr[0], a_arr[0]);
        assert_eq!(ab_arr[1], a_arr[1]);
        assert_eq!(ab_arr[2], b_arr[0]);
        assert_eq!(ab_arr[3], b_arr[1]);
        assert_eq!(ab_arr[4], b_arr[2]);

        let ab_concat = (a, b).concat_tensor_along(Axis::<0>);
        assert_eq!(ab.array(), ab_concat.array());
        let concat_grads = ab_concat.exp().sum().backward();
        let ab_grads = ab.leaky_trace().exp().sum().backward();

        assert_close_to_tensor!(concat_grads.get(&ab), ab_grads.get(&ab));
    }

    #[test]
    fn test_split_ax_1() {
        let dev: TestDevice = Default::default();
        let ab: Tensor<Rank3<2, 5, 4>, TestDtype, _> = dev.sample_normal();
        let ab_dyn = ab
            .leaky_trace()
            .try_realize::<(Const<2>, usize, Const<4>)>()
            .unwrap();
        let (a, b, _tape) = ab_dyn.split_tensor_along(Axis::<1>, 2, 3);
        let a = a.try_realize::<(Const<2>, Const<2>, Const<4>)>().unwrap();
        let b = b.try_realize::<(Const<2>, Const<3>, Const<4>)>().unwrap();
        let ab_arr = ab.array();
        let a_arr = a.array();
        let b_arr = b.array();
        println!("{a_arr:?}");
        println!("{b_arr:?}");
        println!("{ab_arr:?}");

        for i in 0..2 {
            assert_eq!(ab_arr[i][0], a_arr[i][0]);
            assert_eq!(ab_arr[i][1], a_arr[i][1]);
            assert_eq!(ab_arr[i][2], b_arr[i][0]);
            assert_eq!(ab_arr[i][3], b_arr[i][1]);
            assert_eq!(ab_arr[i][4], b_arr[i][2]);
        }

        let ab_concat = (a, b).concat_tensor_along(Axis::<1>);
        assert_eq!(ab.array(), ab_concat.array());
        let concat_grads = ab_concat.exp().sum().backward();
        let ab_grads = ab.leaky_trace().exp().sum().backward();

        println!("{:?}", concat_grads.get(&ab).array());
        println!("{:?}", ab_grads.get(&ab).array());

        assert_close_to_tensor!(concat_grads.get(&ab), ab_grads.get(&ab));
    }

    #[test]
    fn test_split_ax_2() {
        let dev: TestDevice = Default::default();
        let ab: Tensor<Rank3<2, 3, 5>, TestDtype, _> = dev.sample_normal();
        let ab_dyn = ab
            .leaky_trace()
            .try_realize::<(Const<2>, Const<3>, usize)>()
            .unwrap();
        let (a, b, _tape) = ab_dyn.split_tensor_along(Axis::<2>, 2, 3);
        let a = a.try_realize::<(Const<2>, Const<3>, Const<2>)>().unwrap();
        let b = b.try_realize::<(Const<2>, Const<3>, Const<3>)>().unwrap();
        let ab_arr = ab.array();
        let a_arr = a.array();
        let b_arr = b.array();
        println!("{a_arr:?}");
        println!("{b_arr:?}");
        println!("{ab_arr:?}");

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(ab_arr[i][j][0], a_arr[i][j][0]);
                assert_eq!(ab_arr[i][j][1], a_arr[i][j][1]);
                assert_eq!(ab_arr[i][j][2], b_arr[i][j][0]);
                assert_eq!(ab_arr[i][j][3], b_arr[i][j][1]);
                assert_eq!(ab_arr[i][j][4], b_arr[i][j][2]);
            }
        }

        let ab_concat = (a, b).concat_tensor_along(Axis::<2>);
        assert_eq!(ab.array(), ab_concat.array());
        let concat_grads = ab_concat.exp().sum().backward();
        let ab_grads = ab.leaky_trace().exp().sum().backward();

        println!("{:?}", concat_grads.get(&ab).array());
        println!("{:?}", ab_grads.get(&ab).array());

        assert_close_to_tensor!(concat_grads.get(&ab), ab_grads.get(&ab));
    }
}
