use super::base::*;
use crate::{
    gradients::*,
    traits::{ShapedArray, Tensor},
};
use ndarray::prelude::*;

trait MapOp {
    fn forward(f: f32) -> f32;
    fn derivative(f: f32) -> f32;

    fn call<T: Tensor>(tensor: &mut T) -> T {
        let grad = tensor.take_tape().map(|mut tape| {
            let parent_deriv = tape.store_derivative(tensor.data().mapv(Self::derivative));
            let result_grad = tape.store_gradient(T::SHAPE);

            tape.add_operation(Operation::Unary(UnaryOp {
                op_type: OpType::Map,
                parent_grad: tensor.gradient_ref(),
                parent_deriv,
                result_grad,
            }));

            Grad::with_tape(result_grad, tape)
        });

        T::with_grad(tensor.data().mapv(Self::forward), grad)
    }
}

struct ReLU;
impl MapOp for ReLU {
    fn forward(f: f32) -> f32 {
        0.0f32.max(f)
    }

    fn derivative(f: f32) -> f32 {
        if f >= 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

struct Square;
impl MapOp for Square {
    fn forward(f: f32) -> f32 {
        f.powi(2)
    }

    fn derivative(f: f32) -> f32 {
        2.0 * f
    }
}

struct Tanh;
impl MapOp for Tanh {
    fn forward(f: f32) -> f32 {
        f.tanh()
    }

    fn derivative(f: f32) -> f32 {
        1.0 - f.tanh().powi(2)
    }
}

struct Sigmoid;
impl MapOp for Sigmoid {
    fn forward(_f: f32) -> f32 {
        todo!()
    }

    fn derivative(_f: f32) -> f32 {
        todo!()
    }
}

struct Sin;
impl MapOp for Sin {
    fn forward(f: f32) -> f32 {
        f.sin()
    }

    fn derivative(f: f32) -> f32 {
        f.cos()
    }
}

struct Cos;
impl MapOp for Cos {
    fn forward(f: f32) -> f32 {
        f.cos()
    }

    fn derivative(f: f32) -> f32 {
        f.sin()
    }
}

macro_rules! unary_ops {
    ([$($const_defs:tt)*] $typename:ident [$($consts:tt)*], $num_elems:expr) => {
        impl<$($const_defs)*> $typename<$($consts)*> {
            pub fn relu(&mut self) -> Self {
                ReLU::call(self)
            }

            pub fn square(&mut self) -> Self {
                Square::call(self)
            }

            pub fn tanh(&mut self) -> Self {
                Tanh::call(self)
            }

            pub fn sigmoid(&mut self) -> Self {
                Sigmoid::call(self)
            }

            pub fn sin(&mut self) -> Self {
                Sin::call(self)
            }

            pub fn cos(&mut self) -> Self {
                Cos::call(self)
            }

            pub fn mean(&mut self) -> Tensor0D {
                let grad = self.take_tape().map(|mut tape| {
                    let parent_deriv = tape.store_derivative(self.data.mapv(|_f| 1.0 / Self::NUM_ELEMENTS as f32));
                    let result_grad = tape.store_gradient(());

                    tape.add_operation(Operation::Unary(UnaryOp {
                        op_type: OpType::Mean,
                        parent_grad: self.gradient_ref(),
                        parent_deriv,
                        result_grad,
                    }));

                    Grad::with_tape(result_grad, tape)
                });

                Tensor0D {
                    data: arr0(self.data.mean().unwrap()),
                    grad,
                }
            }
        }
    };
}

unary_ops!([] Tensor0D [], 1.0);
unary_ops!([const N: usize] Tensor1D [N], N);
unary_ops!([const M: usize, const N: usize] Tensor2D [M, N], M * N);
