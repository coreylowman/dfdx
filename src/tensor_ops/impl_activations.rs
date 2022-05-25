use crate::prelude::*;
use std::ops::Neg;

/// [Rectified Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) computes `max(0, x)`.
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
    fn f(x: &f32) -> f32 {
        x.max(0.0)
    }

    fn df(x: &f32) -> f32 {
        if x > &0.0 {
            1.0
        } else {
            0.0
        }
    }

    map(t, f, df)
}

/// Square computes `x * x`.
///
/// The derivative is `2 * x`.
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
    fn f(x: &f32) -> f32 {
        x.powi(2)
    }

    fn df(x: &f32) -> f32 {
        2.0 * x
    }

    map(t, f, df)
}

/// Square root computes `x ^ 0.5` or `√x`.
///
/// The derivative is `0.5 / (x ^ 0.5)`.
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
    fn f(x: &f32) -> f32 {
        x.sqrt()
    }

    fn df(x: &f32) -> f32 {
        0.5 * x.sqrt().recip()
    }

    map(t, f, df)
}

/// [Hyperbolic Tangent (Tanh)](https://en.wikipedia.org/wiki/Hyperbolic_functions) computes `tanh(x)`.
///
/// The derivative is `1.0 - square(tanh(x))`.
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
    fn f(x: &f32) -> f32 {
        x.tanh()
    }

    fn df(x: &f32) -> f32 {
        1.0 - x.tanh().powi(2)
    }

    map(t, f, df)
}

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) computes `1 / (1 + exp(-x))`.
///
/// The derivative is `sigmoid(x) * (1.0 - sigmoid(x))`.
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

    fn df(x: &f32) -> f32 {
        let s = f(x);
        s * (1.0 - s)
    }

    map(t, f, df)
}

/// The [sine function](https://en.wikipedia.org/wiki/Sine_and_cosine) computes `sin(x)`
///
/// It's derivative is `cos(x)`
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
    fn f(x: &f32) -> f32 {
        x.sin()
    }
    fn df(x: &f32) -> f32 {
        x.cos()
    }

    map(t, f, df)
}

/// The [cos function](https://en.wikipedia.org/wiki/Sine_and_cosine) computes `cos(x)`
///
/// It's derivative is `-sin(x)`
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
    fn f(x: &f32) -> f32 {
        x.cos()
    }

    fn df(x: &f32) -> f32 {
        x.sin().neg()
    }

    map(t, f, df)
}

/// The [Natural Logarithm (ln)](https://en.wikipedia.org/wiki/Natural_logarithm) computes `ln(x)`
///
/// It's derivative is `1 / x`.
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
    fn f(x: &f32) -> f32 {
        x.ln()
    }

    fn df(x: &f32) -> f32 {
        x.recip()
    }

    map(t, f, df)
}

/// The [exponential function (exp)](https://en.wikipedia.org/wiki/Natural_logarithm) computes `e ^ x`
///
/// It's derivative is itself! `e ^ x`.
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
    fn f(x: &f32) -> f32 {
        x.exp()
    }

    fn df(x: &f32) -> f32 {
        x.exp()
    }

    map(t, f, df)
}

/// The [absolute value (abs)](https://en.wikipedia.org/wiki/Absolute_value) computes `|x|`
///
/// The derivative is -1.0 for x < 0, 0 for x == 0, and 1.0 for x > 0.
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
    fn f(x: &f32) -> f32 {
        x.abs()
    }

    fn df(x: &f32) -> f32 {
        if x == &0.0 {
            0.0
        } else {
            x.signum()
        }
    }

    map(t, f, df)
}

/// Applies a function `f` to every element of the [Tensor]. The derivative
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
pub fn map<T: Tensor<Dtype = f32>, F, DF>(t: T, f: F, mut df: DF) -> T
where
    F: 'static + FnMut(&f32) -> f32 + Copy,
    DF: 'static + FnMut(&f32) -> f32 + Copy,
{
    let result = T::NoTape::new_boxed(T::Device::map(t.data(), f));
    let (mut t, mut tape_holder) = t.split_tape_holder();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        // t = df(t) * result_grad
        T::Device::zip_map_assign(t.mut_data(), tape.ref_gradient(&_result), &mut |l, r| {
            *l = df(l) * *r
        });
        T::Device::add_assign(tape.mut_gradient(&t), t.data());
    });
    result.with_tape_holder(tape_holder)
}

/// Similar to [map()], but doesn't take ownership of the [Tensor] `t`.
pub fn cloned_map<T, F: FnMut(&f32) -> f32 + Copy>(t: &T, f: F) -> T
where
    T: Tensor<Dtype = f32, TapeHolder = NoTape> + TensorCreator,
{
    T::new_boxed(T::Device::map(t.data(), f))
}

macro_rules! activation_impl {
    ($func_name:ident) => {
        /// Calls [$func_name] on `self`
        pub fn $func_name(self) -> Self {
            $func_name(self)
        }
    };
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> $typename<$($Vs, )* H> {
    activation_impl!(relu);
    activation_impl!(sin);
    activation_impl!(cos);
    activation_impl!(ln);
    activation_impl!(exp);
    activation_impl!(sigmoid);
    activation_impl!(tanh);
    activation_impl!(square);
    activation_impl!(sqrt);
    activation_impl!(abs);
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
}
