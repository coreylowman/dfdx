use crate::gradients::GradientTape;
use ndarray::{Array, Dimension, ShapeBuilder};
use rand::{distributions::Distribution, Rng};

pub trait HasUniqueId {
    fn id(&self) -> usize;
}

pub trait CanUpdateWithTape {
    fn update_with_tape(&mut self, tape: &GradientTape);
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

pub trait Tensor: IsShapedArray + CanUpdateWithTape + HasUniqueId {}

pub trait TensorNoTape: Default + Tensor {
    type WithTape: Tensor
        + TensorWithTape<NoTape = Self>
        + IsShapedArray<Dimension = Self::Dimension>;

    fn new_no_tape(data: Array<f32, Self::Dimension>) -> Self;
    fn zeros() -> Self;
    fn ones() -> Self;
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn randn<R: Rng>(rng: &mut R) -> Self;

    fn with_tape(&self) -> Self::WithTape;
    fn put_tape(self, tape: Box<GradientTape>) -> Self::WithTape;
}

pub trait TensorWithTape: Tensor {
    type NoTape: Tensor + TensorNoTape<WithTape = Self> + IsShapedArray<Dimension = Self::Dimension>;

    fn new_with_tape(data: Array<f32, Self::Dimension>, tape: Box<GradientTape>) -> Self;
    fn without_tape(self) -> (Self::NoTape, Box<GradientTape>);
    fn backward(self) -> Box<GradientTape>
    where
        Self: Sized,
    {
        let id = self.id();
        let (_, mut tape) = self.without_tape();
        tape.backward(id);
        tape
    }
}

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}
