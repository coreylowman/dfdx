use super::traits::{Batch, Randomize, Record, ShapedArray, Tensor};
use crate::gradients::{traits::Params, Grad, GradientTape};
use ndarray::prelude::{Array, Ix0, Ix1, Ix2};
use ndarray_rand::rand::{distributions::Distribution, Rng};

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

        impl<$($const_defs)*> Tensor for $typename<$($consts)*> {
            fn with_grad(data: Array<f32, Self::Dimension>, grad: Option<Grad>) -> Self {
                Self { data, grad }
            }

            fn grad(&self) -> &Option<Grad> {
                &self.grad
            }

            fn mut_grad(&mut self) -> &mut Option<Grad> {
                &mut self.grad
            }
        }

        impl<$($const_defs)*> Randomize for $typename<$($consts)*> {
            fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
                self.mut_data().map_inplace(|f| *f = dist.sample(rng))
            }
        }

        impl<$($const_defs)*> Record for $typename<$($consts)*> {
            fn record(&mut self, tape: &mut GradientTape) {
                if self.grad().is_none() {
                    *self.mut_grad() = Some(Grad::new(tape.store_gradient(Self::SHAPE)));
                }
            }
        }

        impl<$($const_defs)*> Params for $typename<$($consts)*> {
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
