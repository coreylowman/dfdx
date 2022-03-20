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

impl TapeManager for WithTape {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, mut update_fn: F) {
        update_fn(&mut self.0)
    }
}

impl TapeManager for NoTape {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, _update_fn: F) {}
}

macro_rules! tensor_impl {
    ($typename:ident, [$($const_names:tt),*]) => {
        impl<$(const $const_names: usize, )* Mgr1: TapeManager, Mgr2: TapeManager> CanAddTape<Mgr2> for $typename<$($const_names, )* Mgr1> {
            type Output = $typename<$($const_names, )* Mgr2>;

            fn with_tape_manager(self, mgr: Mgr2) -> Self::Output {
                Self::Output { id: self.id, data: self.data, tape: mgr }
            }
        }

        impl<$(const $const_names: usize),*> TensorCreator for $typename<$($const_names, )* NoTape> {
            fn new(data: Array<f32, Self::Dimension>) -> Self {
                Self { id: unique_id(), data, tape: NoTape::default() }
            }
        }

        impl<$(const $const_names: usize),*> TapeCreator for $typename<$($const_names, )* NoTape> {
            fn with_tape(&self) -> Self::WithTape {
                Self::WithTape { id: self.id, data: self.data.clone(), tape: WithTape::default() }
            }
        }

        impl<$(const $const_names: usize, )* Mgr: TapeManager> Tensor for $typename<$($const_names, )* Mgr> {
            type TapeManager = Mgr;
            type NoTape = $typename<$($const_names, )* NoTape>;
            type WithTape = $typename<$($const_names, )* WithTape>;

            fn split_tape_manager(self) -> (Self::NoTape, Self::TapeManager) {
                (
                    Self::NoTape { id: self.id, data: self.data, tape: NoTape::default() },
                    self.tape,
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

pub trait TensorSugar {
    fn zeros() -> Self;
    fn ones() -> Self;
    fn rand<R: rand::Rng>(rng: &mut R) -> Self;
    fn randn<R: rand::Rng>(rng: &mut R) -> Self;
}

impl<T: Tensor + TensorCreator> TensorSugar for T {
    fn zeros() -> Self {
        Self::new(Array::zeros(Self::SHAPE))
    }
    fn ones() -> Self {
        Self::new(Array::ones(Self::SHAPE))
    }

    fn rand<R: rand::Rng>(rng: &mut R) -> Self {
        Self::new(Array::from_shape_simple_fn(Self::SHAPE, || {
            rand_distr::Standard.sample(rng)
        }))
    }

    fn randn<R: rand::Rng>(rng: &mut R) -> Self {
        Self::new(Array::from_shape_simple_fn(Self::SHAPE, || {
            rand_distr::StandardNormal.sample(rng)
        }))
    }
}

pub fn backward<T: Tensor<TapeManager = WithTape>>(t: T) -> Box<GradientTape> {
    let id = t.id();
    let (_, mut tape_manager) = t.split_tape_manager();
    tape_manager.0.backward(id);
    tape_manager.0
}
