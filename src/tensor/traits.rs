use super::{NoTape, WithTape};
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

pub trait TapeHolder {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, update_fn: F);
}

pub trait CanReplaceTapeHolder<T: TapeHolder> {
    type Output;
    fn replace_tape_holder(self, tape_holder: T) -> Self::Output;
}

pub trait Tensor: IsShapedArray + CanUpdateWithTape + HasUniqueId
where
    Self: Sized,
{
    type TapeHolder: TapeHolder;
    type NoTape: Tensor<TapeHolder = NoTape, Dimension = Self::Dimension>;
    type WithTape: Tensor<TapeHolder = WithTape, Dimension = Self::Dimension>;
    fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder);
}

pub trait TensorCreator: Tensor {
    fn new(data: Array<f32, Self::Dimension>) -> Self;

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

pub trait TapeCreator: Tensor {
    fn with_tape(&self) -> Self::WithTape;
}

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}
