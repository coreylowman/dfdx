/// A set of differtiable functions specified by structs
/// implementing [DifferentiableFunction] or [DiffBinaryFunction].
///
/// All of the [DifferentiableFunction] also implement [Module] to make it easy to
/// use them in neural networks.

/// A function that acts on 1 value that is differentiable.
pub trait DifferentiableFunction {
    /// The actual function
    fn f(x: f32) -> f32;

    /// The derivative of the function at `x`. dfdx!
    fn df(x: f32) -> f32;
}

/// A function that acts on 2 values that is differentiable
pub trait DiffBinaryFunction {
    // The actual function
    fn f(x: &f32, y: &f32) -> f32;

    // The partial derivative of f wrt x (the first parameter)
    fn dfdx(x: &f32, y: &f32) -> f32;

    // The partial derivative of f wrt y (the second parameter)
    fn dfdy(x: &f32, y: &f32) -> f32;
}

/// [Rectified Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) computes `max(0, x)`.
///
/// The derivative is the [Heaviside](https://en.wikipedia.org/wiki/Heaviside_step_function) function.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// assert_eq!(ReLU::f(1.0), 1.0);
/// assert_eq!(ReLU::f(-1.0), 0.0);
/// assert_eq!(ReLU::df(1.0), 1.0);
/// assert_eq!(ReLU::df(-1.0), 0.0);
/// ```
#[derive(Default, Debug, Clone, Copy)]
pub struct ReLU;
impl DifferentiableFunction for ReLU {
    fn f(x: f32) -> f32 {
        0.0f32.max(x)
    }

    fn df(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
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
/// assert_eq!(Square::f(2.0), 4.0);
/// assert_eq!(Square::f(-2.0), 4.0);
/// assert_eq!(Square::df(1.0), 2.0);
/// assert_eq!(Square::df(-1.0), -2.0);
/// ```
#[derive(Default, Debug, Clone, Copy)]
pub struct Square;
impl DifferentiableFunction for Square {
    fn f(x: f32) -> f32 {
        x.powi(2)
    }

    fn df(x: f32) -> f32 {
        2.0 * x
    }
}

/// [Hyperbolic Tangent (Tanh)](https://en.wikipedia.org/wiki/Hyperbolic_functions) computes `tanh(x)`.
///
/// The derivative is `1.0 - square(tanh(x))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// assert_eq!(Tanh::f(0.0), 0.0);
/// assert_eq!(Tanh::f(1.0), 0.76159415595);
/// assert_eq!(Tanh::df(0.0), 1.0);
/// assert_eq!(Tanh::df(1.0), 0.41997433);
/// ```
#[derive(Default, Debug, Clone, Copy)]
pub struct Tanh;
impl DifferentiableFunction for Tanh {
    fn f(x: f32) -> f32 {
        x.tanh()
    }

    fn df(x: f32) -> f32 {
        1.0 - x.tanh().powi(2)
    }
}

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) computes `1 / (1 + exp(-x))`.
///
/// The derivative is `sigmoid(x) * (1.0 - sigmoid(x))`
#[derive(Default, Debug, Clone, Copy)]
pub struct Sigmoid;
impl DifferentiableFunction for Sigmoid {
    fn f(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn df(x: f32) -> f32 {
        let s = Self::f(x);
        s * (1.0 - s)
    }
}

/// The [sine function](https://en.wikipedia.org/wiki/Sine_and_cosine) computes `sin(x)`
///
/// It's derivative is `cos(x)`
#[derive(Default, Debug, Clone, Copy)]
pub struct Sin;
impl DifferentiableFunction for Sin {
    fn f(x: f32) -> f32 {
        x.sin()
    }
    fn df(x: f32) -> f32 {
        x.cos()
    }
}

/// The [cos function](https://en.wikipedia.org/wiki/Sine_and_cosine) computes `cos(x)`
///
/// It's derivative is `-sin(x)`
#[derive(Default, Debug, Clone, Copy)]
pub struct Cos;
impl DifferentiableFunction for Cos {
    fn f(x: f32) -> f32 {
        x.cos()
    }
    fn df(x: f32) -> f32 {
        -x.sin()
    }
}

/// The [Natural Logarithm (ln)](https://en.wikipedia.org/wiki/Natural_logarithm) computes `ln(x)`
///
/// It's derivative is `1 / x`
#[derive(Default, Debug, Clone, Copy)]
pub struct Ln;
impl DifferentiableFunction for Ln {
    fn f(x: f32) -> f32 {
        x.ln()
    }
    fn df(x: f32) -> f32 {
        1.0 / x
    }
}

/// The [exponential function (exp)](https://en.wikipedia.org/wiki/Natural_logarithm) computes `e ^ x`
///
/// It's derivative is itself! `e ^ x`
#[derive(Default, Debug, Clone, Copy)]
pub struct Exp;
impl DifferentiableFunction for Exp {
    fn f(x: f32) -> f32 {
        x.exp()
    }
    fn df(x: f32) -> f32 {
        x.exp()
    }
}

/// The [absolute value (abs)](https://en.wikipedia.org/wiki/Absolute_value) computes `|x|`
///
/// The derivative is -1.0 for x < 0, 0 for x == 0, and 1.0 for x > 0.
#[derive(Default, Debug, Clone, Copy)]
pub struct Abs;
impl DifferentiableFunction for Abs {
    fn f(x: f32) -> f32 {
        x.abs()
    }

    fn df(x: f32) -> f32 {
        if x < 0.0 {
            -1.0
        } else if x == 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

/// Binary add represents adding two numbers together.
#[derive(Default, Debug, Clone, Copy)]
pub struct BinaryAdd;
impl DiffBinaryFunction for BinaryAdd {
    fn f(x: &f32, y: &f32) -> f32 {
        x + y
    }

    fn dfdx(_: &f32, _: &f32) -> f32 {
        1.0
    }

    fn dfdy(_: &f32, _: &f32) -> f32 {
        1.0
    }
}

/// Binary sub represents subtracting a number from another number.
#[derive(Default, Debug, Clone, Copy)]
pub struct BinarySub;

impl DiffBinaryFunction for BinarySub {
    fn f(x: &f32, y: &f32) -> f32 {
        x - y
    }

    fn dfdx(_: &f32, _: &f32) -> f32 {
        1.0
    }

    fn dfdy(_: &f32, _: &f32) -> f32 {
        -1.0
    }
}

/// Represents multiplying two numbers.
#[derive(Default, Debug, Clone, Copy)]
pub struct BinaryMul;

impl DiffBinaryFunction for BinaryMul {
    fn f(x: &f32, y: &f32) -> f32 {
        x * y
    }

    fn dfdx(_: &f32, y: &f32) -> f32 {
        *y
    }

    fn dfdy(x: &f32, _: &f32) -> f32 {
        *x
    }
}

/// Represents diving a number by another number.
#[derive(Default, Debug, Clone, Copy)]
pub struct BinaryDiv;

impl DiffBinaryFunction for BinaryDiv {
    fn f(x: &f32, y: &f32) -> f32 {
        x / y
    }

    fn dfdx(_x: &f32, y: &f32) -> f32 {
        1.0 / y
    }

    fn dfdy(x: &f32, y: &f32) -> f32 {
        -x / y.powi(2)
    }
}
