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

pub trait TapeManager {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, update_fn: F);
}

pub trait CanAddTape<Mgr: TapeManager> {
    type Output;
    fn with_tape_manager(self, mgr: Mgr) -> Self::Output;
}

pub trait Tensor:
    IsShapedArray + CanUpdateWithTape + HasUniqueId + CanAddTape<WithTape> + CanAddTape<NoTape>
where
    Self: Sized,
{
    type TapeManager: TapeManager;

    type NoTape: Tensor<
        TapeManager = NoTape,
        NoTape = Self::NoTape,
        WithTape = Self::WithTape,
        Dimension = Self::Dimension,
    >;
    type WithTape: Tensor<
        TapeManager = WithTape,
        NoTape = Self::NoTape,
        WithTape = Self::WithTape,
        Dimension = Self::Dimension,
    >;

    fn split_tape_manager(self) -> (Self::NoTape, Self::TapeManager);
}

pub trait TensorCreator: Tensor {
    fn new(data: Array<f32, Self::Dimension>) -> Self;
}

pub trait TapeCreator: Tensor {
    fn with_tape(&self) -> Self::WithTape;
}

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}
