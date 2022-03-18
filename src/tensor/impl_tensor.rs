use super::structs::*;
use super::traits::*;
use crate::prelude::GradientTape;
use ndarray::Array;
use rand::distributions::Distribution;
use std::sync::atomic::{AtomicUsize, Ordering};

fn unique_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

macro_rules! tensor_impl {
    ($typename:ident, [$($const_names:tt),*]) => {
        impl<$(const $const_names: usize, )*> Tensor for $typename<$($const_names, )* NoTape> {}

        impl<$(const $const_names: usize, )*> Tensor for $typename<$($const_names, )* WithTape> {}

        impl<$(const $const_names: usize),*> TensorNoTape for $typename<$($const_names, )* NoTape> {
            type WithTape = $typename<$($const_names, )* WithTape>;

            fn new_no_tape(data: Array<f32, Self::Dimension>) -> Self {
                Self { id: unique_id(), data, tape: NoTape::default() }
            }

            fn with_tape(&self) -> Self::WithTape {
                Self::WithTape { id: self.id, data: self.data.clone(), tape: Default::default() }
            }

            fn put_tape(self, tape: Box<GradientTape>) -> Self::WithTape {
                Self::WithTape { id: self.id, data: self.data, tape: WithTape(tape) }
            }

            fn zeros() -> Self { Self::new_no_tape(Array::zeros(Self::SHAPE)) }
            fn ones() -> Self { Self::new_no_tape(Array::ones(Self::SHAPE)) }

            fn rand<R: rand::Rng>(rng: &mut R) -> Self {
                Self::new_no_tape(Array::from_shape_simple_fn(Self::SHAPE, || rand_distr::Standard.sample(rng)))
            }

            fn randn<R: rand::Rng>(rng: &mut R) -> Self {
                Self::new_no_tape(Array::from_shape_simple_fn(Self::SHAPE, || rand_distr::StandardNormal.sample(rng)))
            }
        }

        impl<$(const $const_names: usize),*> TensorWithTape for $typename<$($const_names, )* WithTape> {
            type NoTape = $typename<$($const_names, )* NoTape>;

            fn new_with_tape(data: Array<f32, Self::Dimension>, tape: Box<GradientTape>) -> Self {
                Self { id: unique_id(), data, tape: WithTape(tape) }
            }

            fn without_tape(self) -> (Self::NoTape, Box<GradientTape>) {
                (
                    Self::NoTape { id: self.id, data: self.data, tape: NoTape::default() },
                    self.tape.0,
                )
            }
        }
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [N]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);
