use crate::prelude::*;

pub trait HasSumLastMethod: Tensor {
    type Output: Tensor;
    fn sum_last(self) -> Self::Output;
}

macro_rules! sum_last_impl {
    ($typename:ident, [$($Vs:tt),*], $res:ident, [$($Zs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> HasSumLastMethod for $typename<$($Vs, )* H> {
    type Output = $res<$($Zs, )* H>;
    fn sum_last(self) -> Self::Output {
        let result = <$res<$($Zs, )* H> as Tensor>::NoTape::new(self.data().reduce_inner(|a, b| a + b));
        let (t, mut tape_holder) = self.split_tape_holder();
        let deriv: Self::ArrayType = t.data().map_elems(|_| 1.0);
        let _t = t.phantom();
        let _result = result.phantom();
        tape_holder.add_operation(move |tape| {
            let g: &<Self::ArrayType as ReduceInnerElements>::Output = tape.gradient(&_result);
            let d_grad: Self::ArrayType = deriv.mul(g);
            tape.mut_gradient(&_t).add_assign(&d_grad);
        });
        result.with_tape_holder(tape_holder)
    }
}
    };
}

sum_last_impl!(Tensor1D, [M], Tensor0D, []);
sum_last_impl!(Tensor2D, [M, N], Tensor1D, [M]);
sum_last_impl!(Tensor3D, [M, N, O], Tensor2D, [M, N]);
sum_last_impl!(Tensor4D, [M, N, O, P], Tensor3D, [M, N, O]);

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_sum_last_1d() {
//         let t: Tensor1D<3> = Tensor1D::new(arr1(&[1.0, 2.0, 3.0]));
//         let r: Tensor1D<1, WithTape> = t.with_tape().sum_last();
//         assert_eq!(r.data(), arr1(&[6.0]));
//         let gradients = backward(r.mean());
//         assert_eq!(
//             gradients
//                 .gradient_for(t.id())
//                 .clone()
//                 .into_shape((3,))
//                 .unwrap(),
//             arr1(&[1.0; 3])
//         );
//     }

//     #[test]
//     fn test_sum_last_2d() {
//         let t: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
//         let r: Tensor2D<2, 1, WithTape> = t.with_tape().sum_last();
//         assert_eq!(r.data(), arr2(&[[6.0], [15.0]]));
//         let gradients = backward(r.mean());
//         assert_eq!(
//             gradients
//                 .gradient_for(t.id())
//                 .clone()
//                 .into_shape((2, 3))
//                 .unwrap(),
//             arr2(&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
//         );
//     }
// }
