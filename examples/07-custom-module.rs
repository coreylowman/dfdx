//! Demonstrates how to build your own [nn::Module]

use dfdx::prelude::*;
use rand::prelude::*;

/// Custom model struct
/// This case is trivial and should be done with a tuple of linears and relus,
/// but it demonstrates how to build models with custom behavior
#[derive(Default)]
struct Mlp<const IN: usize, const INNER: usize, const OUT: usize> {
    l1: Linear<IN, INNER>,
    l2: Linear<INNER, OUT>,
    relu: ReLU,
}

// ResetParams let's you randomize your models parameters
impl<const IN: usize, const INNER: usize, const OUT: usize> ResetParams for Mlp<IN, INNER, OUT> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.l1.reset_params(rng);
        self.l2.reset_params(rng);
        self.relu.reset_params(rng);
    }
}

// CanUpdateWithGradients let's you update your model's parameters with gradients
impl<const IN: usize, const INNER: usize, const OUT: usize> CanUpdateWithGradients
    for Mlp<IN, INNER, OUT>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.l1.update(grads, unused);
        self.l2.update(grads, unused);
        self.relu.update(grads, unused);
    }
}

// impl Module for single item
impl<const IN: usize, const INNER: usize, const OUT: usize> Module<Tensor1D<IN>>
    for Mlp<IN, INNER, OUT>
{
    type Output = Tensor1D<OUT>;

    fn forward(&self, x: Tensor1D<IN>) -> Self::Output {
        let x = self.l1.forward(x);
        let x = self.relu.forward(x);
        self.l2.forward(x)
    }
}

// impl Module for batch of items
impl<const BATCH: usize, const IN: usize, const INNER: usize, const OUT: usize, TAPE: Tape>
    Module<Tensor2D<BATCH, IN, TAPE>> for Mlp<IN, INNER, OUT>
{
    type Output = Tensor2D<BATCH, OUT, TAPE>;

    fn forward(&self, x: Tensor2D<BATCH, IN, TAPE>) -> Self::Output {
        let x = self.l1.forward(x);
        let x = self.relu.forward(x);
        self.l2.forward(x)
    }
}

const BATCH_SIZE: usize = 32;

fn main() {
    // Rng for generating model's params
    let mut rng = StdRng::seed_from_u64(0);

    // Construct model
    let mut model: Mlp<10, 512, 10> = Mlp::default();
    model.reset_params(&mut rng);

    // Forward pass with a single sample
    let sample: Tensor1D<10> = Tensor1D::randn(&mut rng);
    let _: Tensor1D<10> = model.forward(sample);

    // Forward pass with a batch of samples
    let batch: Tensor2D<BATCH_SIZE, 10> = Tensor2D::randn(&mut rng);
    let _: Tensor2D<BATCH_SIZE, 10, OwnedTape> = model.forward(batch.trace());
}
