pub trait DifferentiableFunction {
    fn f(x: f32) -> f32;
    fn df(x: f32) -> f32;
}

pub trait DiffBinaryFunction {
    fn f(x: &f32, y: &f32) -> f32;
    fn dfdx(x: &f32, y: &f32) -> f32;
    fn dfdy(x: &f32, y: &f32) -> f32;
}

#[derive(Default, Debug)]
pub struct ReLU;
impl DifferentiableFunction for ReLU {
    fn f(x: f32) -> f32 {
        0.0f32.max(x)
    }

    fn df(x: f32) -> f32 {
        if x >= 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Default, Debug)]
pub struct Square;
impl DifferentiableFunction for Square {
    fn f(x: f32) -> f32 {
        x.powi(2)
    }

    fn df(x: f32) -> f32 {
        2.0 * x
    }
}

#[derive(Default, Debug)]
pub struct Tanh;
impl DifferentiableFunction for Tanh {
    fn f(x: f32) -> f32 {
        x.tanh()
    }

    fn df(x: f32) -> f32 {
        1.0 - x.tanh().powi(2)
    }
}
#[derive(Default, Debug)]
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
#[derive(Default, Debug)]
pub struct Sin;
impl DifferentiableFunction for Sin {
    fn f(x: f32) -> f32 {
        x.sin()
    }
    fn df(x: f32) -> f32 {
        x.cos()
    }
}

#[derive(Default, Debug)]
pub struct Cos;
impl DifferentiableFunction for Cos {
    fn f(x: f32) -> f32 {
        x.cos()
    }
    fn df(x: f32) -> f32 {
        x.sin()
    }
}

#[derive(Default, Debug)]
pub struct Ln;
impl DifferentiableFunction for Ln {
    fn f(x: f32) -> f32 {
        x.ln()
    }
    fn df(x: f32) -> f32 {
        1.0 / x
    }
}

#[derive(Default, Debug)]
pub struct Exp;
impl DifferentiableFunction for Exp {
    fn f(x: f32) -> f32 {
        x.exp()
    }
    fn df(x: f32) -> f32 {
        x.exp()
    }
}

#[derive(Default, Debug)]
pub struct Abs;
impl DifferentiableFunction for Abs {
    fn f(x: f32) -> f32 {
        x.abs()
    }

    fn df(x: f32) -> f32 {
        if x <= 0.0 {
            -1.0
        } else {
            1.0
        }
    }
}

#[derive(Default, Debug)]
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

#[derive(Default, Debug)]
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

#[derive(Default, Debug)]
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

#[derive(Default, Debug)]
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
