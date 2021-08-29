use crate::gradients::GradientTape;
use crate::traits::{Module, Params, Tensor};
use ndarray_rand::rand::Rng;

pub trait Optimizer<M: Module>: Module {
    fn step<T: Tensor>(&mut self, loss: &mut T);
}

#[derive(Default, Debug)]
pub struct Sgd<M: Module> {
    lr: f32,
    module: M,
}

impl<M: Module> Sgd<M> {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            module: Default::default(),
        }
    }
}

impl<M: Module> Params for Sgd<M> {
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.module.randomize(rng);
    }

    fn register(&mut self, tape: &mut GradientTape) {
        self.module.register(tape);
    }

    fn update(&mut self, tape: &GradientTape) {
        self.module.update(tape);
    }
}

impl<M: Module> Module for Sgd<M> {
    type Input = M::Input;
    type Output = M::Output;

    fn forward(&mut self, input: &mut Self::Input) -> Self::Output {
        let mut tape = GradientTape::new();

        // register module's params
        self.register(&mut tape);

        // register input params
        input.set_tag(Some(tape.advance(input.shape())));

        // put tape in input
        input.keep_tape(Some(Box::new(tape)));

        // go!
        self.module.forward(input)
    }
}

impl<M: Module> Optimizer<M> for Sgd<M> {
    fn step<T: Tensor>(&mut self, loss: &mut T) {
        let mut tape = *loss.backward();
        tape.scale(self.lr);
        self.update(&tape);
    }
}
