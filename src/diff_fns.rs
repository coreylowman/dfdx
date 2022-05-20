use num_traits::Float;

/// A set of differtiable functions specified by structs
/// implementing [DifferentiableFunction] or [DiffBinaryFunction].
///
/// All of the [DifferentiableFunction] also implement [Module] to make it easy to
/// use them in neural networks.

/// A function that acts on 1 value that is differentiable.
pub trait DifferentiableFunction<T> {
    /// The actual function
    fn f(x: &T) -> T;

    /// The derivative of the function at `x`. dfdx!
    fn df(x: &T) -> T;
}

/// A function that acts on 2 values that is differentiable
pub trait DiffBinaryFunction<T> {
    // The actual function
    fn f(x: &T, y: &T) -> T;

    // The partial derivative of f wrt x (the first parameter)
    fn dfdx(x: &T, y: &T) -> T;

    // The partial derivative of f wrt y (the second parameter)
    fn dfdy(x: &T, y: &T) -> T;
}

/// [Rectified Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) computes `max(0, x)`.
///
/// The derivative is the [Heaviside](https://en.wikipedia.org/wiki/Heaviside_step_function) function.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// assert_eq!(ReLU::f(&1.0), 1.0);
/// assert_eq!(ReLU::f(&-1.0), 0.0);
/// assert_eq!(ReLU::df(&1.0), 1.0);
/// assert_eq!(ReLU::df(&-1.0), 0.0);
/// ```
#[derive(Default, Debug, Clone, Copy)]
pub struct ReLU;
impl<T: Float> DifferentiableFunction<T> for ReLU {
    fn f(x: &T) -> T {
        x.max(T::zero())
    }

    fn df(x: &T) -> T {
        if x > &T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }
}

/// Square computes `x * x`.
///
/// The derivative is `2 * x`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// assert_eq!(Square::f(&2.0), 4.0);
/// assert_eq!(Square::f(&-2.0), 4.0);
/// assert_eq!(Square::df(&1.0), 2.0);
/// assert_eq!(Square::df(&-1.0), -2.0);
/// ```
#[derive(Default, Debug, Clone, Copy)]
pub struct Square;
impl<T: Float> DifferentiableFunction<T> for Square {
    fn f(x: &T) -> T {
        x.powi(2)
    }

    fn df(x: &T) -> T {
        *x + *x
    }
}

/// [Hyperbolic Tangent (Tanh)](https://en.wikipedia.org/wiki/Hyperbolic_functions) computes `tanh(x)`.
///
/// The derivative is `1.0 - square(tanh(x))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// assert_eq!(Tanh::f(&0.0), 0.0);
/// assert_eq!(Tanh::f(&1.0f32), 0.76159415595);
/// assert_eq!(Tanh::df(&0.0), 1.0);
/// assert_eq!(Tanh::df(&1.0f32), 0.41997433);
/// ```
#[derive(Default, Debug, Clone, Copy)]
pub struct Tanh;
impl<T: Float> DifferentiableFunction<T> for Tanh {
    fn f(x: &T) -> T {
        x.tanh()
    }

    fn df(x: &T) -> T {
        T::one() - x.tanh().powi(2)
    }
}

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) computes `1 / (1 + exp(-x))`.
///
/// The derivative is `sigmoid(x) * (1.0 - sigmoid(x))`
#[derive(Default, Debug, Clone, Copy)]
pub struct Sigmoid;
impl<T: Float> DifferentiableFunction<T> for Sigmoid {
    fn f(x: &T) -> T {
        (T::one() + x.neg().exp()).recip()
    }

    fn df(x: &T) -> T {
        let s = Self::f(x);
        s * (T::one() - s)
    }
}

/// The [sine function](https://en.wikipedia.org/wiki/Sine_and_cosine) computes `sin(x)`
///
/// It's derivative is `cos(x)`
#[derive(Default, Debug, Clone, Copy)]
pub struct Sin;
impl<T: Float> DifferentiableFunction<T> for Sin {
    fn f(x: &T) -> T {
        x.sin()
    }
    fn df(x: &T) -> T {
        x.cos()
    }
}

/// The [cos function](https://en.wikipedia.org/wiki/Sine_and_cosine) computes `cos(x)`
///
/// It's derivative is `-sin(x)`
#[derive(Default, Debug, Clone, Copy)]
pub struct Cos;
impl<T: Float> DifferentiableFunction<T> for Cos {
    fn f(x: &T) -> T {
        x.cos()
    }
    fn df(x: &T) -> T {
        x.sin().neg()
    }
}

/// The [Natural Logarithm (ln)](https://en.wikipedia.org/wiki/Natural_logarithm) computes `ln(x)`
///
/// It's derivative is `1 / x`
#[derive(Default, Debug, Clone, Copy)]
pub struct Ln;
impl<T: Float> DifferentiableFunction<T> for Ln {
    fn f(x: &T) -> T {
        x.ln()
    }
    fn df(x: &T) -> T {
        x.recip()
    }
}

/// The [exponential function (exp)](https://en.wikipedia.org/wiki/Natural_logarithm) computes `e ^ x`
///
/// It's derivative is itself! `e ^ x`
#[derive(Default, Debug, Clone, Copy)]
pub struct Exp;
impl<T: Float> DifferentiableFunction<T> for Exp {
    fn f(x: &T) -> T {
        x.exp()
    }
    fn df(x: &T) -> T {
        x.exp()
    }
}

/// The [absolute value (abs)](https://en.wikipedia.org/wiki/Absolute_value) computes `|x|`
///
/// The derivative is -1.0 for x < 0, 0 for x == 0, and 1.0 for x > 0.
///
/// ```rust
/// # use dfdx::prelude::*;
/// assert_eq!(Abs::f(&2.0), 2.0);
/// assert_eq!(Abs::f(&0.0), 0.0);
/// assert_eq!(Abs::f(&-2.0), 2.0);
/// assert_eq!(Abs::df(&2.0), 1.0);
/// assert_eq!(Abs::df(&0.0), 0.0);
/// assert_eq!(Abs::df(&-2.0), -1.0);
/// ```
#[derive(Default, Debug, Clone, Copy)]
pub struct Abs;
impl<T: Float> DifferentiableFunction<T> for Abs {
    fn f(x: &T) -> T {
        x.abs()
    }

    fn df(x: &T) -> T {
        if x == &T::zero() {
            T::zero()
        } else {
            x.signum()
        }
    }
}

/// Binary add represents adding two numbers together.
#[derive(Default, Debug, Clone, Copy)]
pub struct BinaryAdd;
impl<T: Float> DiffBinaryFunction<T> for BinaryAdd {
    fn f(x: &T, y: &T) -> T {
        *x + *y
    }

    fn dfdx(_: &T, _: &T) -> T {
        T::one()
    }

    fn dfdy(_: &T, _: &T) -> T {
        T::one()
    }
}

/// Binary sub represents subtracting a number from another number.
#[derive(Default, Debug, Clone, Copy)]
pub struct BinarySub;

impl<T: Float> DiffBinaryFunction<T> for BinarySub {
    fn f(x: &T, y: &T) -> T {
        *x - *y
    }

    fn dfdx(_: &T, _: &T) -> T {
        T::one()
    }

    fn dfdy(_: &T, _: &T) -> T {
        T::one().neg()
    }
}

/// Represents multiplying two numbers.
#[derive(Default, Debug, Clone, Copy)]
pub struct BinaryMul;

impl<T: Float> DiffBinaryFunction<T> for BinaryMul {
    fn f(x: &T, y: &T) -> T {
        *x * *y
    }

    fn dfdx(_: &T, y: &T) -> T {
        *y
    }

    fn dfdy(x: &T, _: &T) -> T {
        *x
    }
}

/// Represents diving a number by another number.
#[derive(Default, Debug, Clone, Copy)]
pub struct BinaryDiv;

impl<T: Float> DiffBinaryFunction<T> for BinaryDiv {
    fn f(x: &T, y: &T) -> T {
        *x * y.recip()
    }

    fn dfdx(_x: &T, y: &T) -> T {
        y.recip()
    }

    fn dfdy(x: &T, y: &T) -> T {
        x.neg() * y.powi(2).recip()
    }
}
