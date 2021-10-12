use super::tensor::{Batch, InitSugar, Record, ShapedArray, Tensor};
use crate::gradients::{Gradient, GradientTape, HasGradient, Taped};
use ndarray::prelude::{Array, Ix0, Ix1, Ix2};

#[derive(Debug)]
pub struct Tensor0D {
    pub(super) data: Array<f32, Ix0>,
    pub(super) grad: Option<Gradient>,
}

#[derive(Debug)]
pub struct Tensor1D<const N: usize> {
    pub(super) data: Array<f32, Ix1>,
    pub(super) grad: Option<Gradient>,
}

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize> {
    pub(super) data: Array<f32, Ix2>,
    pub(super) grad: Option<Gradient>,
}

macro_rules! tensor_impl {
    ([$($const_defs:tt)*] $typename:ident [$($consts:tt)*], $dim:ty, $shape_val:expr, $shape_type:ty, $num_elems:expr) => {
        impl<$($const_defs)*> ShapedArray for $typename<$($consts)*> {
            type Dimension = $dim;
            type Shape = $shape_type;
            const SHAPE: Self::Shape = $shape_val;
            const NUM_ELEMENTS: usize = $num_elems;

            fn data(&self) -> &Array<f32, Self::Dimension> {
                &self.data
            }

            fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
                &mut self.data
            }
        }

        impl<$($const_defs)*> Default for $typename<$($consts)*> {
            fn default() -> Self {
                Self {
                    data: Array::zeros(Self::SHAPE),
                    grad: None,
                }
            }
        }

        impl<$($const_defs)*> InitSugar for $typename<$($consts)*> { }

        impl<$($const_defs)*> HasGradient for $typename<$($consts)*> {
            fn grad(&self) -> &Option<Gradient> {
                &self.grad
            }

            fn mut_grad(&mut self) -> &mut Option<Gradient> {
                &mut self.grad
            }
        }

        impl<$($const_defs)*> Tensor for $typename<$($consts)*> { }

        impl<$($const_defs)*> Record for $typename<$($consts)*> {
            fn record(&mut self, tape: &mut GradientTape) {
                if self.grad().is_none() {
                    *self.mut_grad() = Some(Gradient::new(tape.register_gradient(Self::SHAPE)));
                }
            }
        }

        impl<$($const_defs)*> Taped for $typename<$($consts)*> {
            fn update(&mut self, tape: &GradientTape) {
                let grad = self.mut_grad().take().unwrap();
                *self.mut_data() -= &tape[grad.gradient_ref];
            }
        }
    }
}

tensor_impl!([] Tensor0D [], Ix0, (), (), 1);
tensor_impl!([const N: usize] Tensor1D [N], Ix1, (N,), (usize,), N);
tensor_impl!([const M: usize, const N: usize] Tensor2D [M, N], Ix2, (M, N), (usize, usize), M * N);

impl Batch for Tensor0D {
    type Batched<const B: usize> = Tensor1D<B>;
}

impl<const N: usize> Batch for Tensor1D<N> {
    type Batched<const B: usize> = Tensor2D<B, N>;
}
