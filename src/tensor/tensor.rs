use crate::gradients::{ops::GradientRef, traits::Taped, Gradient, GradientTape};
use ndarray::{Array, Dimension, ShapeBuilder};
use ndarray_rand::{
    rand::{distributions::Distribution, Rng},
    rand_distr::{Standard, StandardNormal},
};

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}

pub trait ShapedArray {
    type Dimension: Dimension;
    type Shape: ShapeBuilder<Dim = Self::Dimension>;
    const SHAPE: Self::Shape;
    const NUM_ELEMENTS: usize;

    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;
}

pub trait InitSugar: Default + ShapedArray {
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

pub trait Tensor: Randomize + Taped + Default + ShapedArray + Activations + InitSugar {
    fn grad(&self) -> &Option<Gradient>;
    fn mut_grad(&mut self) -> &mut Option<Gradient>;

    fn gradient_ref(&self) -> GradientRef {
        self.grad().as_ref().unwrap().gradient_ref
    }

    fn take_tape(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad()
            .as_mut()
            .map(|grad| grad.tape.take())
            .flatten()
    }

    fn backward(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad().as_mut().map(|grad| {
            let mut tape = grad.tape.take().unwrap();
            tape.backward(grad.gradient_ref);
            tape
        })
    }

    fn keep_tape(&mut self, mut tape: Box<GradientTape>) {
        let grad = self
            .mut_grad()
            .get_or_insert_with(|| Gradient::new(tape.store_gradient(Self::SHAPE)));
        grad.tape = Some(tape);
    }
}

pub trait Batch {
    type Batched<const B: usize>: Tensor;
}

pub(super) trait Record {
    fn record(&mut self, tape: &mut GradientTape);
}
