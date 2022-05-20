use crate::prelude::*;

/// Applies a [DifferentiableFunction] to the tensor.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let r = apply::<Tensor1D<5>, ReLU>(t);
/// assert_eq!(r.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
/// ```
///
/// All the differentiable functions are also provided as methods on all tensors:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let r = t.relu();
/// assert_eq!(r.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
/// ```
pub fn apply<T: Tensor, F: DifferentiableFunction<f32>>(t: T) -> T {
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), F::f));
    let (mut t, mut tape_holder) = t.split_tape_holder();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        // t = F::df(t) * result_grad
        T::Device::zip_map_assign(t.mut_data(), tape.ref_gradient(&_result), |l, r| {
            *l = F::df(l) * r
        });
        T::Device::add_assign(tape.mut_gradient(&t), t.data());
    });
    result.with_tape_holder(tape_holder)
}

pub fn apply_ref<T, F: DifferentiableFunction<f32>>(t: &T) -> T
where
    T: Tensor<TapeHolder = NoTape> + TensorCreator,
{
    T::new_boxed(T::Device::map(t.data(), F::f))
}

macro_rules! apply_impl {
    ($trait_name:ident, $method_name:ident, $activation_struct:ident) => {
        pub trait $trait_name {
            fn $method_name(self) -> Self;
        }

        impl<T: Tensor> $trait_name for T {
            fn $method_name(self) -> Self {
                apply::<Self, $activation_struct>(self)
            }
        }
    };
}

apply_impl!(HasReLUMethod, relu, ReLU);
apply_impl!(HasSinMethod, sin, Sin);
apply_impl!(HasCosMethod, cos, Cos);
apply_impl!(HasLnMethod, ln, Ln);
apply_impl!(HasExpMethod, exp, Exp);
apply_impl!(HasSigmoidMethod, sigmoid, Sigmoid);
apply_impl!(HasTanhMethod, tanh, Tanh);
apply_impl!(HasSquareMethod, square, Square);
apply_impl!(HasAbsMethod, abs, Abs);

macro_rules! apply_ref_impl {
    ($trait_name:ident, $method_name:ident, $activation_struct:ident) => {
        pub trait $trait_name {
            fn $method_name(&self) -> Self;
        }

        impl<T: Tensor<TapeHolder = NoTape> + TensorCreator> $trait_name for T {
            fn $method_name(&self) -> Self {
                apply_ref::<Self, $activation_struct>(self)
            }
        }
    };
}

apply_ref_impl!(HasReLURefMethod, relu_, ReLU);
apply_ref_impl!(HasSinRefMethod, sin_, Sin);
apply_ref_impl!(HasCosRefMethod, cos_, Cos);
apply_ref_impl!(HasLnRefMethod, ln_, Ln);
apply_ref_impl!(HasExpRefMethod, exp_, Exp);
apply_ref_impl!(HasSigmoidRefMethod, sigmoid_, Sigmoid);
apply_ref_impl!(HasTanhRefMethod, tanh_, Tanh);
apply_ref_impl!(HasSquareRefMethod, square_, Square);
apply_ref_impl!(HasAbsRefMethod, abs_, Abs);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().relu();
        assert_eq!(r.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&x), &[0.0, 0.0, 0.0, 0.2, 0.2]);
    }

    #[test]
    fn test_sin() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sin();
        assert_eq!(
            r.data(),
            &[-0.90929741, -0.84147096, 0.00000000, 0.84147096, 0.90929741]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[-0.08322937, 0.10806046, 0.20000000, 0.10806046, -0.08322937]
        );
    }

    #[test]
    fn test_cos() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().cos();
        assert_eq!(
            r.data(),
            &[-0.41614684, 0.5403023, 1.0, 0.5403023, -0.41614684]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[0.18185948, 0.16829419, -0.0, -0.16829419, -0.18185948]
        );
    }

    #[test]
    fn test_ln() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().ln();
        assert!(r.data()[0].is_nan());
        assert!(r.data()[1].is_nan());
        assert!(&r.data()[2..] == &[f32::NEG_INFINITY, 0.0, 0.69314718]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[-0.1, -0.2, f32::INFINITY, 0.2, 0.1]
        );
    }

    #[test]
    fn test_exp() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().exp();
        assert_eq!(
            r.data(),
            &[0.13533528, 0.36787945, 1.0, 2.71828175, 7.38905621]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[0.027067056, 0.07357589, 0.2, 0.54365635, 1.47781122]
        );
    }

    #[test]
    fn test_sigmoid() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sigmoid();
        assert_eq!(
            r.data(),
            &[0.11920292, 0.26894143, 0.50000000, 0.73105860, 0.88079703]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[0.020998716, 0.039322387, 0.05, 0.039322387, 0.020998726]
        );
    }

    #[test]
    fn test_tanh() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().tanh();
        assert_eq!(
            r.data(),
            &[-0.96402758, -0.76159418, 0.00000000, 0.76159418, 0.96402758]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[0.014130163, 0.083994865, 0.2, 0.083994865, 0.014130163]
        );
    }

    #[test]
    fn test_square() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().square();
        assert_eq!(r.data(), &[4.0, 1.0, 0.0, 1.0, 4.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&x), &[-0.8, -0.4, 0.0, 0.4, 0.8]);
    }

    #[test]
    fn test_abs() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().abs();
        assert_eq!(r.data(), &[2.0, 1.0, 0.0, 1.0, 2.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&x), &[-0.2, -0.2, 0.0, 0.2, 0.2]);
    }
}
