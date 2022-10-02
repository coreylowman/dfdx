use crate::prelude::*;
use rand::Rng;

macro_rules! activation_impls {
    ($struct_name:ident, $func_name:ident, #[$docstring:meta]) => {
        #[$docstring]
        #[derive(Default, Debug, Clone, Copy)]
        pub struct $struct_name;

        impl CanUpdateWithGradients for $struct_name {
            /// Does nothing.
            fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
        }

        impl ResetParams for $struct_name {
            /// Does nothing.
            fn reset_params<R: Rng>(&mut self, _: &mut R) {}
        }

        impl SaveToNpz for $struct_name {}
        impl LoadFromNpz for $struct_name {}

        impl<T: Tensor<Dtype = f32>> Module<T> for $struct_name {
            type Output = T;
            fn forward(&self, input: T) -> Self::Output {
                $func_name(input)
            }
        }
    };
}

activation_impls!(ReLU, relu, #[doc="Unit struct that impls [Module] as calling [relu()] on `input`."]);
activation_impls!(Sin, sin, #[doc="Unit struct that impls [Module] as calling [sin()] on `input`."]);
activation_impls!(Cos, cos, #[doc="Unit struct that impls [Module] as calling [cos()] on `input`."]);
activation_impls!(Ln, ln, #[doc="Unit struct that impls [Module] as calling [ln()] on `input`."]);
activation_impls!(Exp, exp, #[doc="Unit struct that impls [Module] as calling [exp()] on `input`."]);
activation_impls!(Sigmoid, sigmoid, #[doc="Unit struct that impls [Module] as calling [sigmoid()] on `input`."]);
activation_impls!(Tanh, tanh, #[doc="Unit struct that impls [Module] as calling [tanh()] on `input`."]);
activation_impls!(Square, square, #[doc="Unit struct that impls [Module] as calling [square()] on `input`."]);
activation_impls!(Sqrt, sqrt, #[doc="Unit struct that impls [Module] as calling [sqrt()] on `input`."]);
activation_impls!(Abs, abs, #[doc="Unit struct that impls [Module] as calling [abs()] on `input`."]);

/// Unit struct that impls [Module] as calling [softmax()] on `input`."
#[derive(Default, Debug, Clone, Copy)]
pub struct Softmax;

impl CanUpdateWithGradients for Softmax {
    /// Does nothing.
    fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
}

impl ResetParams for Softmax {
    /// Does nothing.
    fn reset_params<R: Rng>(&mut self, _: &mut R) {}
}

impl SaveToNpz for Softmax {}
impl LoadFromNpz for Softmax {}

impl<T> Module<T> for Softmax
where
    T: Reduce<<<T as HasArrayType>::Array as HasLastAxis>::LastAxis>,
    T::Array: HasLastAxis,
{
    type Output = T;
    fn forward(&self, input: T) -> Self::Output {
        softmax(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = ReLU.forward(t.clone());
        let r2 = relu(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_sin() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Sin.forward(t.clone());
        let r2 = sin(t);
        assert_eq!(r1.data(), r2.data());
    }
    #[test]
    fn test_cos() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Cos.forward(t.clone());
        let r2 = cos(t);
        assert_eq!(r1.data(), r2.data());
    }
    #[test]
    fn test_ln() {
        let t = tensor([0.0, 1.0, 2.0, 3.0, 4.0]);
        let r1 = Ln.forward(t.clone());
        let r2 = ln(t);
        assert_eq!(r1.data(), r2.data());
    }
    #[test]
    fn test_exp() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Exp.forward(t.clone());
        let r2 = exp(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_sigmoid() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Sigmoid.forward(t.clone());
        let r2 = sigmoid(t);
        assert_eq!(r1.data(), r2.data());
    }
    #[test]
    fn test_tanh() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Tanh.forward(t.clone());
        let r2 = tanh(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_square() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Square.forward(t.clone());
        let r2 = square(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_sqrt() {
        let t = tensor([0.0, 1.0, 2.0, 3.0, 4.0]);
        let r1 = Sqrt.forward(t.clone());
        let r2 = sqrt(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_abs() {
        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Abs.forward(t.clone());
        let r2 = abs(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_softmax() {
        let t = Tensor0D::new(0.0);
        let r1 = Softmax.forward(t.clone());
        let r2 = softmax(t);
        assert_eq!(r1.data(), r2.data());

        let t = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Softmax.forward(t.clone());
        let r2 = softmax(t);
        assert_eq!(r1.data(), r2.data());

        let t = Tensor2D::new([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);
        let r1 = Softmax.forward(t.clone());
        let r2 = t.softmax::<1>();
        assert_eq!(r1.data(), r2.data());
    }
}
