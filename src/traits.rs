use ndarray::{Array, Dimension};
use ndarray_rand::rand::prelude::*;

use crate::gradients::{GradientRef, GradientTape};

pub trait Params {
    fn randomize<R: Rng>(&mut self, rng: &mut R);
    fn register(&mut self, tape: &mut GradientTape);
    fn update(&mut self, tape: &GradientTape);
}

pub trait Tensor: Params + Default {
    type Dimension: Dimension;
    const SHAPE: &'static [usize];

    fn grad(&self) -> &GradientRef;
    fn mut_grad(&mut self) -> &mut GradientRef;
    fn data(&self) -> &Array<f32, Self::Dimension>;
    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension>;

    fn backward(&mut self) -> Box<GradientTape> {
        self.mut_grad().backward()
    }

    fn set_tag(&mut self, tag: Option<usize>) {
        self.mut_grad().set_tag(tag);
    }

    fn take_tape(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad().take_tape()
    }

    fn keep_tape(&mut self, tape: Option<Box<GradientTape>>) {
        self.mut_grad().keep_tape(tape);
    }
}

impl<T> Params for T
where
    T: Tensor,
{
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.mut_data().map_inplace(|f| *f = rng.gen())
    }

    fn register(&mut self, tape: &mut GradientTape) {
        if !self.grad().has_tag() {
            self.set_tag(Some(tape.advance(Self::SHAPE)));
        }
    }

    fn update(&mut self, tape: &GradientTape) {
        let gradient = &tape[self.grad().tag()];
        *self.mut_data() -= gradient;
        self.set_tag(None);
    }
}

pub trait Module: Params + Default {
    type Input: Tensor;
    type Output: Tensor;

    fn forward(&mut self, input: &mut Self::Input) -> Self::Output;
}
