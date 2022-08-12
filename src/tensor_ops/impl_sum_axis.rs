use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;

pub fn sum_axis<T: Tensor<Dtype = f32> + Reduce1<I>, const I: isize>(t: T) -> T::Reduced {
    let mut result = <T::Reduced as Tensor>::NoTape::zeros();
    T::DeviceR::reduce_into(t.data(), result.mut_data(), |a, b| a + b);
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::DeviceR::foreach_br(t_grad, result_grad, &mut |l, r| {
            *l += r;
        })
    })
}

macro_rules! sum_axis_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [sum_axis()] on `self`.
    pub fn sum_axis<const I: isize>(self) -> <Self as Reduce1<I>>::Reduced
    where
        Self: Reduce1<I>
    {
        sum_axis::<Self, I>(self)
    }
}
    };
}

sum_axis_impl!(Tensor0D, []);
sum_axis_impl!(Tensor1D, [M]);
sum_axis_impl!(Tensor2D, [M, N]);
sum_axis_impl!(Tensor3D, [M, N, O]);
sum_axis_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valids_sum_axis() {
        let _: Tensor0D = Tensor1D::<5>::zeros().sum_axis::<0>();
        let _: Tensor0D = Tensor1D::<5>::zeros().sum_axis::<-1>();

        let _: Tensor1D<3> = Tensor2D::<5, 3>::zeros().sum_axis::<0>();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().sum_axis::<1>();
        let _: Tensor1D<5> = Tensor2D::<5, 3>::zeros().sum_axis::<-1>();

        let _: Tensor2D<5, 3> = Tensor3D::<7, 5, 3>::zeros().sum_axis::<0>();
        let _: Tensor2D<7, 3> = Tensor3D::<7, 5, 3>::zeros().sum_axis::<1>();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().sum_axis::<2>();
        let _: Tensor2D<7, 5> = Tensor3D::<7, 5, 3>::zeros().sum_axis::<-1>();

        let _: Tensor3D<7, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<0>();
        let _: Tensor3D<9, 5, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<1>();
        let _: Tensor3D<9, 7, 3> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<2>();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<3>();
        let _: Tensor3D<9, 7, 5> = Tensor4D::<9, 7, 5, 3>::zeros().sum_axis::<-1>();
    }

    #[test]
    fn test_sum_axis_last() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<OwnedTape> = t.trace().sum_axis::<-1>();
        assert_eq!(r.data(), &6.0);
        // NOTE: .exp() so we make sure its using result grad properly
        let gradients = r.exp().mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[403.4288; 3]);
    }
}
