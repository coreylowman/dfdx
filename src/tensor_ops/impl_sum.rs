use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

/// Sums all the values in `self`. Returns a [Tensor0D] (i.e. one number).
pub fn sum<T: Tensor<Dtype = f32>>(t: T) -> Tensor0D<T::Tape> {
    let result = Tensor0D::<NoneTape>::new(T::Device::reduce(t.data(), &mut |a, b| a + b));
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::Device::foreach_m(t_grad, &mut |v| *v += result_grad);
    })
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [sum()] on `self`.
    pub fn sum(self) -> Tensor0D<<Self as Tensor>::Tape> {
        sum(self)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_0d() {
        let t: Tensor0D = Tensor0D::new(3.0);
        let r = t.trace().sum();
        assert_eq!(r.data(), &3.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_sum_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().sum();
        assert_eq!(r.data(), &6.0);
        // NOTE: .exp() to make sure its using result grad properly
        let gradients = r.exp().backward();
        assert_eq!(gradients.ref_gradient(&t), &[403.4288; 3]);
    }

    #[test]
    fn test_sum_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r: Tensor0D<OwnedTape> = t.trace().sum();
        assert_eq!(r.data(), &21.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &[[1.0; 3]; 2]);
    }

    #[test]
    fn test_sum_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::ones();
        let r: Tensor0D<OwnedTape> = t.trace().sum();
        assert_eq!(r.data(), &(4.0 * 2.0 * 3.0));
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&t), &[[[1.0; 3]; 2]; 4]);
    }
}
