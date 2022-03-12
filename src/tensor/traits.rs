use super::structs::{GradientData, Tensor0D};
use crate::gradients::{GradientRef, GradientTape};
use ndarray::{Array, Dimension, ShapeBuilder};
use rand::{distributions::Distribution, Rng};
use std::cell::RefCell;

pub trait OnGradientTape {
    fn update_with(&mut self, tape: &GradientTape);
}

pub trait HasGradientData {
    fn grad_data(&self) -> &RefCell<GradientData>;
    fn take_tape(&self) -> Option<Box<GradientTape>> {
        let mut grad_data = self.grad_data().borrow_mut();
        grad_data.tape.take()
    }
}

pub trait IsShapedArray {
    type Dimension: Dimension;
    type Shape: ShapeBuilder<Dim = Self::Dimension>;
    const SHAPE: Self::Shape;
    const NUM_ELEMENTS: usize;

    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;
}

pub trait Tensor: Default + IsShapedArray + HasGradientData + OnGradientTape {
    fn new(data: Array<f32, Self::Dimension>, grad_data: GradientData) -> Self;

    fn trace_gradients(&self) {
        let mut grad_data = self.grad_data().borrow_mut();
        grad_data.tape = Some(Box::new(GradientTape::new()));
        grad_data.grad_ref = None;
    }

    fn grad_ref(&self, tape: &mut GradientTape) -> GradientRef {
        let mut grad_data = self.grad_data().borrow_mut();
        // assert!(grad_data.grad_ref.is_none());
        // todo!("lazy allocation of grad ref, so we can overwrite here instead of reusing");
        let grad_ref = grad_data
            .grad_ref
            .get_or_insert_with(|| tape.allocate_gradient(Self::SHAPE));
        *grad_ref
    }

    fn backward(&self) -> Option<GradientTape> {
        let ref_grad_data = self.grad_data();
        let mut grad_data = ref_grad_data.borrow_mut();
        match (grad_data.grad_ref.take(), grad_data.tape.take()) {
            (Some(grad_ref), Some(mut tape)) => {
                tape.backward(grad_ref);
                Some(*tape)
            }
            _ => None,
        }
    }
}

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}

pub trait Mean {
    fn mean(&self) -> Tensor0D;
}

pub trait TensorSugar {
    fn zeros() -> Self;
    fn ones() -> Self;
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn randn<R: Rng>(rng: &mut R) -> Self;

    fn relu(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn ln(&self) -> Self;
    fn exp(&self) -> Self;
    fn sigmoid(&self) -> Self;
    fn tanh(&self) -> Self;
    fn square(&self) -> Self;
    fn abs(&self) -> Self;
}
