use super::utils::move_tape_and_add_backward_op;
use crate::prelude::*;
use std::ops::Neg;

/// Negates all elements.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a: Tensor1D<3> = Tensor1D::new([-2.0, 0.0, 5.0]);
/// let r = -a; // or negate(a);
/// assert_eq!(r.data(), &[2.0, 0.0, -5.0]);
/// ```
pub fn negate<T: Tensor<Dtype = f32>>(t: T) -> T {
    map_df_uses_fx(t, |x| -x, |_| -1.0)
}

/// [Rectified Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). `max(0, t)`
///
/// The derivative is the [Heaviside](https://en.wikipedia.org/wiki/Heaviside_step_function) function.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = relu(t.clone());
///
/// // or the tensor method!
/// let r2 = t.relu();
/// ```
pub fn relu<T: Tensor<Dtype = f32>>(t: T) -> T {
    map_df_uses_fx(t, |x| x.max(0.0), |fx| if fx > &0.0 { 1.0 } else { 0.0 })
}

/// `t^2`
///
/// The derivative is `2 * t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = square(t.clone());
///
/// // or the tensor method!
/// let r2 = t.square();
/// ```
pub fn square<T: Tensor<Dtype = f32>>(t: T) -> T {
    map(t, |x| x.powi(2), |x| 2.0 * x)
}

/// `√t` or `t^0.5`
///
/// The derivative is `0.5 / (t ^ 0.5)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = sqrt(t.clone());
///
/// // or the tensor method!
/// let r2 = t.sqrt();
/// ```
pub fn sqrt<T: Tensor<Dtype = f32>>(t: T) -> T {
    map_df_uses_fx(t, |x| x.sqrt(), |fx| 0.5 * fx.recip())
}

/// [Hyperbolic Tangent (Tanh)](https://en.wikipedia.org/wiki/Hyperbolic_functions).
///
/// The derivative is `1.0 - square(tanh(t))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = tanh(t.clone());
///
/// // or the tensor method!
/// let r2 = t.tanh();
/// ```
pub fn tanh<T: Tensor<Dtype = f32>>(t: T) -> T {
    map_df_uses_fx(t, |x| x.tanh(), |fx| 1.0 - fx.powi(2))
}

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function).
///
/// Equivalent to `1 / (1 + exp(-t))`.
///
/// The derivative is `sigmoid(t) * (1.0 - sigmoid(t))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = sigmoid(t.clone());
///
/// // or the tensor method!
/// let r2 = t.sigmoid();
/// ```
pub fn sigmoid<T: Tensor<Dtype = f32>>(t: T) -> T {
    fn f(x: &f32) -> f32 {
        (1.0 + x.neg().exp()).recip()
    }

    map_df_uses_fx(t, f, |fx| fx * (1.0 - fx))
}

/// [Sine function](https://en.wikipedia.org/wiki/Sine_and_cosine).
///
/// It's derivative is `cos(t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = sin(t.clone());
///
/// // or the tensor method!
/// let r2 = t.sin();
/// ```
pub fn sin<T: Tensor<Dtype = f32>>(t: T) -> T {
    map(t, |x| x.sin(), |x| x.cos())
}

/// [Cosine function](https://en.wikipedia.org/wiki/Sine_and_cosine).
///
/// It's derivative is `-sin(t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = cos(t.clone());
///
/// // or the tensor method!
/// let r2 = t.cos();
/// ```
pub fn cos<T: Tensor<Dtype = f32>>(t: T) -> T {
    map(t, |x| x.cos(), |x| x.sin().neg())
}

/// [Natural Logarithm (ln)](https://en.wikipedia.org/wiki/Natural_logarithm). `log_e(t)`.
///
/// It's derivative is `1 / t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = ln(t.clone());
///
/// // or the tensor method!
/// let r2 = t.ln();
/// ```
pub fn ln<T: Tensor<Dtype = f32>>(t: T) -> T {
    map(t, |x| x.ln(), |x| x.recip())
}

/// [Exponential function (exp)](https://en.wikipedia.org/wiki/Natural_logarithm). `e^t`
///
/// It's derivative is itself! `e^t`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = exp(t.clone());
///
/// // or the tensor method!
/// let r2 = t.exp();
/// ```
pub fn exp<T: Tensor<Dtype = f32>>(t: T) -> T {
    map_df_uses_fx(t, |x| x.exp(), |fx| *fx)
}

/// [Absolute value (abs)](https://en.wikipedia.org/wiki/Absolute_value). `|t|`
///
/// The derivative is -1.0 for t < 0, 0 for t == 0, and 1.0 for t > 0.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-1.0, 0.0, 1.0, 2.0]);
///
/// // use function version
/// let r = abs(t.clone());
///
/// // or the tensor method!
/// let r2 = t.abs();
/// ```
pub fn abs<T: Tensor<Dtype = f32>>(t: T) -> T {
    map(t, |x| x.abs(), |x| if x == &0.0 { 0.0 } else { x.signum() })
}

/// `f(t)`. Applies a function `f` to every element of the [Tensor]. The derivative
/// `df` must also be provided.
///
/// This is primarily used to implement standard functions such as [relu()], [exp()], etc.
/// But users can also implement their own activations with this.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let r = map(t, |x| 2.0 * x, |x| 2.0);
/// assert_eq!(r.data(), &[-4.0, -2.0, 0.0, 2.0, 4.0]);
/// ```
pub fn map<T: Tensor<Dtype = f32>, F, Df>(t: T, f: F, mut df: Df) -> T
where
    F: 'static + FnMut(&f32) -> f32,
    Df: 'static + FnMut(&f32) -> f32,
{
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), f));
    move_tape_and_add_backward_op(t, result, move |t, result, grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        T::Device::foreach_mrr(t_grad, t.data(), result_grad, &mut |g, t, r| {
            *g += df(t) * r;
        });
    })
}

/// Same as [map()], but calls `df` with the result of `f(x)`. This can potentially remove an allocation.
pub fn map_df_uses_fx<T: Tensor<Dtype = f32>, F, Df>(mut t: T, mut f: F, mut df: Df) -> T
where
    F: FnMut(&f32) -> f32,
    Df: 'static + FnMut(&f32) -> f32,
{
    T::Device::foreach_m(t.mut_data(), &mut |x| *x = f(x)); // clones if there is more than 1 reference to t
    let (t, mut tape) = t.split_tape();
    let result = t.clone(); // will always a new reference to t, not start a new one
    let phantom_result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (t_grad, result_grad) = grads.mut_and_ref(&t, &phantom_result);
        T::Device::foreach_mrr(t_grad, t.data(), result_grad, &mut |g, fx, r| {
            *g += df(fx) * r;
        });
    });
    result.put_tape(tape)
}

macro_rules! activation_impl {
    ($func_name:ident, #[$docstring:meta]) => {
        #[$docstring]
        pub fn $func_name(self) -> Self {
            $func_name(self)
        }
    };
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    activation_impl!(negate, #[doc="Calls [negate()] on `self`."]);
    activation_impl!(relu, #[doc="Calls [relu()] on `self`."]);
    activation_impl!(sin, #[doc="Calls [sin()] on `self`."]);
    activation_impl!(cos, #[doc="Calls [cos()] on `self`."]);
    activation_impl!(ln, #[doc="Calls [ln()] on `self`."]);
    activation_impl!(exp, #[doc="Calls [exp()] on `self`."]);
    activation_impl!(sigmoid, #[doc="Calls [sigmoid()] on `self`."]);
    activation_impl!(tanh, #[doc="Calls [tanh()] on `self`."]);
    activation_impl!(square, #[doc="Calls [square()] on `self`."]);
    activation_impl!(sqrt, #[doc="Calls [sqrt()] on `self`."]);
    activation_impl!(abs, #[doc="Calls [abs()] on `self`."]);
}

impl<$(const $Vs: usize, )* H: Tape> std::ops::Neg for $typename<$($Vs, )* H>
{
    type Output = Self;
    /// Calls [negate()] on `self`.
    fn neg(self) -> Self::Output {
        negate(self)
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
    use crate::tests::assert_close;

    use super::*;

    #[test]
    fn test_relu() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().relu();
        assert_eq!(r.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
        // NOTE: call .exp() to make sure we cover cases where .relu() uses the result's gradient
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[0.0, 0.0, 0.0, 0.54365635, 1.4778112]
        );
    }

    #[test]
    fn test_sin() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sin();
        assert_close(
            r.data(),
            &[-0.9092974, -0.84147096, 0.0, 0.84147096, 0.9092974],
        );
        let gradients = r.mean().backward();
        assert_close(
            gradients.ref_gradient(&x),
            &[-0.08322937, 0.10806046, 0.2, 0.10806046, -0.08322937],
        );
    }

    #[test]
    fn test_cos() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().cos();
        assert_close(
            r.data(),
            &[-0.41614684, 0.5403023, 1.0, 0.5403023, -0.41614684],
        );
        let gradients = r.mean().backward();
        assert_close(
            gradients.ref_gradient(&x),
            &[0.18185948, 0.16829419, -0.0, -0.16829419, -0.18185948],
        );
    }

    #[test]
    fn test_ln() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().ln();
        assert!(r.data()[0].is_nan());
        assert!(r.data()[1].is_nan());
        assert!(r.data()[2..] == [f32::NEG_INFINITY, 0.0, std::f32::consts::LN_2]);
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
            &[0.13533528, 0.36787945, 1.0, std::f32::consts::E, 7.389056]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[0.027067056, 0.07357589, 0.2, 0.54365635, 1.4778112]
        );
    }

    #[test]
    fn test_sigmoid() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().sigmoid();
        assert_eq!(
            r.data(),
            &[0.11920292, 0.26894143, 0.5, 0.7310586, 0.880797]
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
            &[-0.9640276, -0.7615942, 0., 0.7615942, 0.9640276]
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
    fn test_sqrt() {
        let x = Tensor1D::new([-1.0, 0.0, 1.0, 4.0]);
        let r = x.trace().sqrt();
        assert!(r.data()[0].is_nan());
        assert_eq!(r.data()[1..], [0.0, 1.0, 2.0]);
        let gradients = r.mean().backward();
        assert!(gradients.ref_gradient(&x)[0].is_nan());
        assert_eq!(
            gradients.ref_gradient(&x)[1..],
            [f32::INFINITY, 0.5 / 4.0, 0.25 / 4.0]
        );
    }

    #[test]
    fn test_abs() {
        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.trace().abs();
        assert_eq!(r.data(), &[2.0, 1.0, 0.0, 1.0, 2.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&x), &[-0.2, -0.2, 0.0, 0.2, 0.2]);
    }

    #[test]
    fn test_0d_neg() {
        let a = Tensor0D::new(10.0);
        let r = -(a.trace());
        assert_eq!(r.data(), &-10.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &-1.0);
    }

    #[test]
    fn test_1d_neg() {
        let a: Tensor1D<3> = Tensor1D::new([-2.0, 0.0, 5.0]);
        let r = -(a.trace());
        assert_eq!(r.data(), &[2.0, 0.0, -5.0]);
        // NOTE: .exp() so we can make sure neg is using result grad properly
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[-2.463019, -0.33333334, -0.0022459824]
        );
    }

    #[test]
    fn test_2d_neg() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[-2.0, 0.0, 5.0], [1.0, 2.0, 3.0]]);
        let r = -(a.trace());
        assert_eq!(r.data(), &[[2.0, 0.0, -5.0], [-1.0, -2.0, -3.0]]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[-1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_3d_neg() {
        let a: Tensor3D<4, 2, 3> = Tensor3D::ones();
        let r = -(a.trace());
        assert_eq!(r.data(), &[[[-1.0; 3]; 2]; 4]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[[-1.0 / 24.0; 3]; 2]; 4]);
    }
}
