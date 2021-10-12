use crate::gradients::{GradientTape, HasGradient, Taped};
use ndarray::{Array, Dimension, ShapeBuilder};
use ndarray_rand::{
    rand::{distributions::Distribution, Rng},
    rand_distr::{Standard, StandardNormal},
};

pub trait ShapedArray {
    type Dimension: Dimension;
    type Shape: ShapeBuilder<Dim = Self::Dimension>;
    const SHAPE: Self::Shape;
    const NUM_ELEMENTS: usize;

    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;
}

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}

impl<T> Randomize for T
where
    T: ShapedArray,
{
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        self.mut_data().map_inplace(|f| *f = dist.sample(rng))
    }
}

pub trait InitSugar {
    fn zeros() -> Self;
    fn ones() -> Self;
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn randn<R: Rng>(rng: &mut R) -> Self;
}

impl<T> InitSugar for T
where
    T: Default + ShapedArray,
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
}

pub trait Activations {
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

pub trait Tensor:
    Randomize + Taped + Default + ShapedArray + Activations + InitSugar + HasGradient
{
}

pub trait Batch {
    type Batched<const B: usize>: Tensor;
}

pub(super) trait Record {
    fn record(&mut self, tape: &mut GradientTape);
}
