use super::structs::*;
use super::traits::*;
use crate::diff_fns::*;
use crate::gradients::{GradientRef, GradientTape};
use ndarray::{Array, Ix0, Ix1, Ix2, Ix3, Ix4};
use rand::prelude::{Distribution, Rng};
use rand_distr::{Standard, StandardNormal};
use std::ops::SubAssign;

macro_rules! prod {
    () => {
        1
    };
    ($head:ident) => {
        $head
    };
    ($head:ident, $($tail:ident),+) => {
        $head * prod!($($tail),+)
    };
}

macro_rules! tupleify {
    () => {
        ()
    };
    ($elem:tt) => {
        ($elem,)
    };
    ($($elems:tt),+) => {
        ($($elems),*)
    };
}

macro_rules! tensor_impl {
    ($typename:ident, [$($const_names:tt),*], $dim:ty, $shape:ty) => {
        impl<$(const $const_names: usize),*> IsShapedArray for $typename<$($const_names),*> {
            type Dimension = $dim;
            type Shape = $shape;
            const SHAPE: Self::Shape = tupleify!($($const_names),*);
            const NUM_ELEMENTS: usize = prod!($($const_names),*);

            fn data(&self) -> &Array<f32, Self::Dimension> { &self.data }
            fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> { &mut self.data }
        }

        impl<$(const $const_names: usize),*> Default for $typename<$($const_names),*> {
            fn default() -> Self {
                Self {
                    data: Array::zeros(Self::SHAPE),
                    grad: None,
                    tape: None,
                }
            }
        }

        impl<$(const $const_names: usize),*> HasGradientTape for $typename<$($const_names),*> {
            fn tape(&self) -> &Option<Box<GradientTape>> { &self.tape }
            fn mut_tape(&mut self) -> &mut Option<Box<GradientTape>> { &mut self.tape }
        }

        impl<$(const $const_names: usize),*> HasGradientRef for $typename<$($const_names),*> {
            fn grad_ref(&self) -> &Option<GradientRef> { &self.grad }
            fn mut_grad_ref(&mut self) -> &mut Option<GradientRef> { &mut self.grad }
        }

        impl<$(const $const_names: usize),*> OnGradientTape for $typename<$($const_names),*> {
            fn put_on(&mut self, tape: &mut GradientTape) {
                if self.grad_ref().is_none() {
                    *self.mut_grad_ref() = Some(tape.allocate_gradient(Self::SHAPE));
                }
            }

            fn update_with(&mut self, tape: &GradientTape) {
                let grad = self.mut_grad_ref().take().unwrap();
                self.mut_data().sub_assign(&tape[grad]);
            }
        }

        impl<$(const $const_names: usize),*> Randomize for $typename<$($const_names),*>
        {
            fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
                self.mut_data().map_inplace(|f| *f = dist.sample(rng))
            }
        }

        impl<$(const $const_names: usize),*> Tensor for $typename<$($const_names),*> {
            fn new(data: Array<f32, Self::Dimension>, grad: Option<GradientRef>, tape: Option<Box<GradientTape>>) -> Self {
                Self { data, grad, tape }
            }
        }
    }
}

tensor_impl!(Tensor0D, [], Ix0, ());
tensor_impl!(Tensor1D, [M], Ix1, (usize,));
tensor_impl!(Tensor2D, [M, N], Ix2, (usize, usize));
tensor_impl!(Tensor3D, [M, N, O], Ix3, (usize, usize, usize));
tensor_impl!(Tensor4D, [M, N, O, P], Ix4, (usize, usize, usize, usize));

impl<T> TensorSugar for T
where
    T: Tensor,
{
    fn zeros() -> Self {
        let mut a = Self::default();
        a.mut_data().fill(0.0);
        a
    }

    fn ones() -> Self {
        let mut a = Self::default();
        a.mut_data().fill(1.0);
        a
    }

    fn rand<R: Rng>(rng: &mut R) -> Self {
        let mut a = Self::default();
        a.mut_data().map_inplace(|f| *f = Standard.sample(rng));
        a
    }

    fn randn<R: Rng>(rng: &mut R) -> Self {
        let mut a = Self::default();
        a.mut_data()
            .map_inplace(|f| *f = StandardNormal.sample(rng));
        a
    }

    fn relu(&mut self) -> Self {
        self.apply::<ReLU>()
    }

    fn sin(&mut self) -> Self {
        self.apply::<Sin>()
    }

    fn cos(&mut self) -> Self {
        self.apply::<Cos>()
    }

    fn ln(&mut self) -> Self {
        self.apply::<Ln>()
    }

    fn exp(&mut self) -> Self {
        self.apply::<Exp>()
    }

    fn sigmoid(&mut self) -> Self {
        self.apply::<Sigmoid>()
    }

    fn tanh(&mut self) -> Self {
        self.apply::<Tanh>()
    }

    fn square(&mut self) -> Self {
        self.apply::<Square>()
    }

    fn abs(&mut self) -> Self {
        self.apply::<Abs>()
    }
}
