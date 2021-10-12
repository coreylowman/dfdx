use super::tensor_impl::{Tensor0D, Tensor1D, Tensor2D};
use crate::gradients::{ops::*, Grad};
use crate::tensor::{Activations, ShapedArray, Tensor};
use ndarray::prelude::*;

mod relu {
    pub(super) fn forward(f: f32) -> f32 {
        0.0f32.max(f)
    }

    pub(super) fn derivative(f: f32) -> f32 {
        if f >= 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

mod square {
    pub(super) fn forward(f: f32) -> f32 {
        f.powi(2)
    }

    pub(super) fn derivative(f: f32) -> f32 {
        2.0 * f
    }
}

mod tanh {
    pub(super) fn forward(f: f32) -> f32 {
        f.tanh()
    }

    pub(super) fn derivative(f: f32) -> f32 {
        1.0 - f.tanh().powi(2)
    }
}

mod sigmoid {
    pub(super) fn forward(f: f32) -> f32 {
        1.0 / (1.0 + (-f).exp())
    }

    pub(super) fn derivative(f: f32) -> f32 {
        let s = forward(f);
        s * (1.0 - s)
    }
}

mod sin {
    pub(super) fn forward(f: f32) -> f32 {
        f.sin()
    }

    pub(super) fn derivative(f: f32) -> f32 {
        f.cos()
    }
}

mod cos {
    pub(super) fn forward(f: f32) -> f32 {
        f.cos()
    }

    pub(super) fn derivative(f: f32) -> f32 {
        f.sin()
    }
}

mod ln {
    pub(super) fn forward(f: f32) -> f32 {
        f.ln()
    }

    pub(super) fn derivative(f: f32) -> f32 {
        1.0 / f
    }
}

mod exp {
    pub(super) fn forward(f: f32) -> f32 {
        f.exp()
    }

    pub(super) fn derivative(f: f32) -> f32 {
        f.exp()
    }
}

mod abs {
    pub(super) fn forward(f: f32) -> f32 {
        f.abs()
    }

    pub(super) fn derivative(f: f32) -> f32 {
        if f <= 0.0 {
            -1.0
        } else {
            1.0
        }
    }
}

macro_rules! map_op_method {
    ($fn_name:ident) => {
        fn $fn_name(&mut self) -> Self {
            let grad = self.take_tape().map(|mut tape| {
                let parent_deriv = tape.store_derivative(self.data().mapv($fn_name::derivative));
                let result_grad = tape.store_gradient(Self::SHAPE);

                tape.add_operation(Operation::Unary(UnaryOp {
                    op_type: OpType::Normal,
                    parent_grad: self.gradient_ref(),
                    parent_deriv,
                    result_grad,
                }));

                Grad::with_tape(result_grad, tape)
            });

            Self {
                data: self.data().mapv($fn_name::forward),
                grad,
            }
        }
    };
}

macro_rules! unary_ops {
    ([$($const_defs:tt)*] $typename:ident [$($consts:tt)*], $num_elems:expr) => {
        impl<$($const_defs)*> Activations for $typename<$($consts)*> {
            map_op_method!(relu);
            map_op_method!(tanh);
            map_op_method!(sigmoid);
            map_op_method!(ln);
            map_op_method!(exp);
            map_op_method!(square);
            map_op_method!(sin);
            map_op_method!(cos);
            map_op_method!(abs);
        }

        impl<$($const_defs)*> $typename<$($consts)*> {
            pub fn mean(&mut self) -> Tensor0D {
                let grad = self.take_tape().map(|mut tape| {
                    let parent_deriv = tape.store_derivative(self.data.mapv(|_f| 1.0 / Self::NUM_ELEMENTS as f32));
                    let result_grad = tape.store_gradient(());

                    tape.add_operation(Operation::Unary(UnaryOp {
                        op_type: OpType::Normal,
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
