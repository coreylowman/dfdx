use crate::prelude::*;
use std::ops::{Add, Mul, Sub};

// MxN + 1xN
pub fn broadcast_outer_add<const M: usize, const N: usize, H: TapeHolder>(
    lhs: Tensor2D<M, N, H>,
    rhs: &Tensor1D<N, NoTape>,
) -> Tensor2D<M, N, H> {
    let mut result = [[0.0; N]; M];
    for i in 0..M {
        result[i] = lhs.data()[i].add(rhs.data());
    }
    let result = Tensor2D::new(result);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let lhs_deriv = lhs.data().map_elems(|_| 1.0);
    let rhs_deriv = rhs.data().map_elems(|_| 1.0);
    let _rhs = rhs.phantom();
    let _lhs = lhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad_lhs = lhs_deriv.mul(tape.gradient(&_result));
        tape.mut_gradient(&_lhs).add_assign(&d_grad_lhs);

        // TODO test this
        let mut d_grad_rhs = [0.0; N];
        for i in 0..M {
            d_grad_rhs.add_assign(&rhs_deriv.mul(&tape.gradient(&_result)[i]));
        }
        tape.mut_gradient(&_rhs).add_assign(&d_grad_rhs);
    });
    result.with_tape_holder(tape_holder)
}

impl<const M: usize, const N: usize, H: TapeHolder> Add<&Tensor1D<N, NoTape>>
    for Tensor2D<M, N, H>
{
    type Output = Tensor2D<M, N, H>;
    fn add(self, rhs: &Tensor1D<N, NoTape>) -> Self::Output {
        broadcast_outer_add(self, rhs)
    }
}

macro_rules! broadcast_sub_impl {
    ($typename:ident, [$($Vs:tt),*], $rhsty:ident, [$($Zs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> std::ops::Sub<&$rhsty<$($Zs, )* NoTape>> for $typename<$($Vs, )* H> {
    type Output = Self;
    fn sub(self, rhs: &$rhsty<$($Zs, )* NoTape>) -> Self::Output {
        let result = <Self::Output as Tensor>::NoTape::new(self.data().sub(rhs.data()));
        let (lhs, mut tape_holder) = self.split_tape_holder();
        let lhs_deriv = lhs.data().map_elems(|_| 1.0);
        let rhs_deriv = rhs.data().map_elems(|_| -1.0);
        let _lhs = lhs.phantom();
        let _rhs = rhs.phantom();
        let _result = result.phantom();
        tape_holder.add_operation(move |tape| {
            let d_grad_lhs = lhs_deriv.mul(tape.gradient(&_result));
            tape.mut_gradient(&_lhs).add_assign(&d_grad_lhs);

            // TODO test this
            let d_grad_rhs = tape.gradient(&_result).mul(&rhs_deriv).reduce_inner(|x, y| x + y);
            tape.mut_gradient(&_rhs).add_assign(&d_grad_rhs);
        });
        result.with_tape_holder(tape_holder)
    }
}
    };
}

broadcast_sub_impl!(Tensor1D, [M], Tensor0D, []);
broadcast_sub_impl!(Tensor2D, [M, N], Tensor1D, [M]);
broadcast_sub_impl!(Tensor3D, [M, N, O], Tensor2D, [M, N]);
broadcast_sub_impl!(Tensor4D, [M, N, O, P], Tensor3D, [M, N, O]);

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_broadcast_sub_1d() {
//         let a: Tensor1D<3> = Tensor1D::new(arr1(&[1.0, 2.0, 3.0]));
//         let b: Tensor1D<1> = Tensor1D::new(arr1(&[1.0]));
//         let r = a.trace() - &b;
//         assert_eq!(r.data(), arr1(&[0.0, 1.0, 2.0]));
//         let gradients = backward(r.mean());
//         assert_eq!(
//             gradients
//                 .gradient_for(a.id())
//                 .clone()
//                 .to_shape((3,))
//                 .unwrap(),
//             arr1(&[1.0 / 3.0; 3])
//         );
//         assert_eq!(
//             gradients
//                 .gradient_for(b.id())
//                 .clone()
//                 .to_shape((1,))
//                 .unwrap(),
//             arr1(&[-1.0; 1])
//         );
//     }

//     #[test]
//     fn test_broadcast_sub_2d() {
//         let a: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
//         let b: Tensor2D<2, 1> = Tensor2D::new(arr2(&[[1.0], [2.0]]));
//         // let r = broadcast_sub_2d(a.trace(), &b);
//         let r = a.trace() - &b;
//         assert_eq!(r.data(), arr2(&[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]));
//         let gradients = backward(r.mean());
//         assert_eq!(
//             gradients
//                 .gradient_for(a.id())
//                 .clone()
//                 .to_shape((2, 3))
//                 .unwrap(),
//             arr2(&[[1.0 / 6.0; 3]; 2])
//         );
//         assert_eq!(
//             gradients
//                 .gradient_for(b.id())
//                 .clone()
//                 .to_shape((2, 1))
//                 .unwrap(),
//             arr2(&[[-0.5; 1]; 2])
//         );
//     }
// }
