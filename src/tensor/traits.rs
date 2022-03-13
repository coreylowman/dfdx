use super::structs::Tensor0D;
use crate::gradients::GradientTape;
use ndarray::{Array, Dimension, ShapeBuilder};
use rand::{distributions::Distribution, Rng};
use std::cell::RefCell;

pub trait OnGradientTape {
    fn update_with(&mut self, tape: &GradientTape);
}

pub trait HasUniqueId {
    fn id(&self) -> usize;
}

pub trait CanStoreGradientTape {
    fn tape(&self) -> &RefCell<Option<Box<GradientTape>>>;
    fn take_tape(&self) -> Option<Box<GradientTape>> {
        self.tape().borrow_mut().take()
    }
    fn put_tape(&self, tape: Option<Box<GradientTape>>) {
        let mut ref_tape = self.tape().borrow_mut();
        *ref_tape = tape;
    }
}

pub trait IsShapedArray {
    type Dimension: Dimension;
    type Shape: ShapeBuilder<Dim = Self::Dimension>;
    const SHAPE: Self::Shape;
    const NUM_ELEMENTS: usize;

    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;

    fn shape(&self) -> Self::Shape {
        Self::SHAPE
    }
}

pub trait Tensor:
    Default + IsShapedArray + CanStoreGradientTape + OnGradientTape + HasUniqueId
{
    fn new(data: Array<f32, Self::Dimension>) -> Self;

    fn trace_gradients(&self) {
        *self.tape().borrow_mut() = Some(Box::new(GradientTape::new()));
    }

    fn backward(&self) -> Option<GradientTape> {
        self.take_tape().map(|mut tape| {
            tape.backward(self.id());
            *tape
        })
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
