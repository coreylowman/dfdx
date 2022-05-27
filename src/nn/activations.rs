use crate::prelude::*;
use rand::{distributions::Distribution, Rng};

macro_rules! activation_impls {
    ($struct_name:ident, $func_name:ident, #[$docstring:meta]) => {
        #[$docstring]
        #[derive(Default, Debug, Clone, Copy)]
        pub struct $struct_name;

        impl CanUpdateWithGradients for $struct_name {
            fn update<G: GradientProvider>(&mut self, _: &mut G) {}
        }

        impl Randomize<f32> for $struct_name {
            fn randomize<R: Rng, D: Distribution<f32>>(&mut self, _: &mut R, _: &D) {}
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = ReLU.forward(t.clone());
        let r2 = relu(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_sin() {
        let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Sin.forward(t.clone());
        let r2 = sin(t);
        assert_eq!(r1.data(), r2.data());
    }
    #[test]
    fn test_cos() {
        let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Cos.forward(t.clone());
        let r2 = cos(t);
        assert_eq!(r1.data(), r2.data());
    }
    #[test]
    fn test_ln() {
        let t = Tensor1D::new([0.0, 1.0, 2.0, 3.0, 4.0]);
        let r1 = Ln.forward(t.clone());
        let r2 = ln(t);
        assert_eq!(r1.data(), r2.data());
    }
    #[test]
    fn test_exp() {
        let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Exp.forward(t.clone());
        let r2 = exp(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Sigmoid.forward(t.clone());
        let r2 = sigmoid(t);
        assert_eq!(r1.data(), r2.data());
    }
    #[test]
    fn test_tanh() {
        let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Tanh.forward(t.clone());
        let r2 = tanh(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_square() {
        let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Square.forward(t.clone());
        let r2 = square(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_sqrt() {
        let t = Tensor1D::new([0.0, 1.0, 2.0, 3.0, 4.0]);
        let r1 = Sqrt.forward(t.clone());
        let r2 = sqrt(t);
        assert_eq!(r1.data(), r2.data());
    }

    #[test]
    fn test_abs() {
        let t = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r1 = Abs.forward(t.clone());
        let r2 = abs(t);
        assert_eq!(r1.data(), r2.data());
    }
}
