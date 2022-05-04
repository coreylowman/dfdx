use crate::prelude::*;

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> std::ops::Neg for $typename<$($Vs, )* H> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let result = <Self::Output as Tensor>::NoTape::new(self.data().map_elems(|v| -v));
        let (t, mut tape_holder) = self.split_tape_holder();
        let _t = t.phantom();
        let _result = result.phantom();
        let deriv = t.data().map_elems(|_| -1.0);
        tape_holder.add_operation(move |tape| {
            let d_grad = deriv.mul(tape.gradient(&_result));
            tape.mut_gradient(&_t).add_assign(&d_grad);
        });
        result.with_tape_holder(tape_holder)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use ndarray::prelude::*;
//     // use std::ops::Neg;

//     #[test]
//     fn test_0d_neg() {
//         let a = Tensor0D::new(arr0(10.0));
//         let gradients = backward(-a.with_tape());
//         assert_eq!(
//             gradients.gradient_for(a.id()).to_shape(a.shape()).unwrap(),
//             arr0(-1.0)
//         );
//     }

//     #[test]
//     fn test_1d_neg() {
//         let a: Tensor1D<3> = Tensor1D::new(arr1(&[-2.0, 0.0, 5.0]));
//         let gradients = backward((-a.with_tape()).mean());
//         assert_eq!(
//             gradients.gradient_for(a.id()).to_shape(a.shape()).unwrap(),
//             arr1(&[-1.0 / 3.0; 3])
//         );
//     }

//     #[test]
//     fn test_2d_neg() {
//         let a: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]]));
//         let gradients = backward((-a.with_tape()).mean());
//         assert_eq!(
//             gradients.gradient_for(a.id()).to_shape(a.shape()).unwrap(),
//             arr2(&[[-1.0 / 6.0; 3]; 2])
//         );
//     }
// }
