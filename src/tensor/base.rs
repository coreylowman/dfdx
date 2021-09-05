use crate::{
    gradients::*,
    traits::{Params, RandomInit, ShapedArray, Tensor},
};
use ndarray::prelude::*;
use ndarray_rand::rand::Rng;

#[derive(Debug)]
pub struct Tensor0D {
    pub(super) data: Array<f32, Ix0>,
    pub(super) grad: Option<Grad>,
}

#[derive(Debug)]
pub struct Tensor1D<const N: usize> {
    pub(super) data: Array<f32, Ix1>,
    pub(super) grad: Option<Grad>,
}

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize> {
    pub(super) data: Array<f32, Ix2>,
    pub(super) grad: Option<Grad>,
}

macro_rules! tensor_impl {
    ([$($parmbounds:tt)*] $typename:ident [$($parm:tt)*], $dim:ty, $shape_val:expr, $shape_type:ty) => {
        impl<$($parmbounds)*> ShapedArray for $typename<$($parm)*> {
            type Dimension = $dim;
            type Shape = $shape_type;
            const SHAPE: Self::Shape = $shape_val;

            fn data(&self) -> &Array<f32, Self::Dimension> {
                &self.data
            }

            fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
                &mut self.data
            }
        }

        impl<$($parmbounds)*> Default for $typename<$($parm)*> {
            fn default() -> Self {
                Self {
                    data: Array::zeros(Self::SHAPE),
                    grad: None,
                }
            }
        }

        impl<$($parmbounds)*> Tensor for $typename<$($parm)*> {
            fn grad(&self) -> &Option<Grad> {
                &self.grad
            }

            fn mut_grad(&mut self) -> &mut Option<Grad> {
                &mut self.grad
            }
        }

        impl<$($parmbounds)*> RandomInit for $typename<$($parm)*> {
            fn randomize<R: Rng>(&mut self, rng: &mut R) {
                self.mut_data().map_inplace(|f| *f = rng.gen())
            }
        }


        impl<$($parmbounds)*> Params for $typename<$($parm)*> {
            fn register(&mut self, tape: &mut GradientTape) {
                if self.grad().is_none() {
                    *self.mut_grad() = Some(Grad::new(tape.store_gradient(Self::SHAPE)));
                }
            }

            fn update(&mut self, tape: &GradientTape) {
                let grad = self.mut_grad().take().unwrap();
                *self.mut_data() -= &tape[grad.gradient_ref];
            }
        }
    }
}

tensor_impl!([] Tensor0D [], Ix0, (), ());
tensor_impl!([const N: usize] Tensor1D [N], Ix1, (N,), (usize,));
tensor_impl!([const M: usize, const N: usize] Tensor2D [M, N], Ix2, (M, N), (usize, usize));
