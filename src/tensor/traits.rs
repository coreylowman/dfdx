use super::structs::Tensor0D;
use crate::{
    gradients::{Gradient, HasGradient},
    prelude::OnGradientTape,
};
use ndarray::{Array, Dimension, ShapeBuilder};
use rand::{distributions::Distribution, Rng};

pub trait IsShapedArray {
    type Dimension: Dimension;
    type Shape: ShapeBuilder<Dim = Self::Dimension>;
    const SHAPE: Self::Shape;
    const NUM_ELEMENTS: usize;

    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;
}

pub trait Tensor: Default + IsShapedArray + HasGradient + OnGradientTape {
    fn new(data: Array<f32, Self::Dimension>, grad: Option<Gradient>) -> Self;
}

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}

pub trait Mean {
    fn mean(&mut self) -> Tensor0D;
}

pub trait TensorSugar {
    fn zeros() -> Self;
    fn ones() -> Self;
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn randn<R: Rng>(rng: &mut R) -> Self;

    fn relu(&mut self) -> Self;
    fn sin(&mut self) -> Self;
    fn cos(&mut self) -> Self;
    fn ln(&mut self) -> Self;
    fn exp(&mut self) -> Self;
    fn sigmoid(&mut self) -> Self;
    fn tanh(&mut self) -> Self;
    fn square(&mut self) -> Self;
    fn abs(&mut self) -> Self;
}
