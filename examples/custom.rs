use dfdx::prelude::*;
use rand::prelude::{SeedableRng, StdRng};

/// Custom model struct
/// This case is trivial and should be done with a tuple of linears and relus,
/// but it demonstrates how to build models with custom behavior
#[derive(Default)]
struct Mlp<const IN: usize, const INNER: usize, const OUT: usize> {
    l1: Linear<IN, INNER>,
    l2: Linear<INNER, OUT>,
    relu: ReLU,
}

impl<const IN: usize, const INNER: usize, const OUT: usize> ResetParams for Mlp<IN, INNER, OUT> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.l1.reset_params(rng);
        self.l2.reset_params(rng);
        self.relu.reset_params(rng);
    }
}

impl<const IN: usize, const INNER: usize, const OUT: usize> CanUpdateWithGradients
    for Mlp<IN, INNER, OUT>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, missing: &mut UnchangedTensors) {
        self.l1.update(grads, missing);
        self.l2.update(grads, missing);
        self.relu.update(grads, missing);
    }
}

// Impl module for single forward pass
impl<const IN: usize, const INNER: usize, const OUT: usize> Module<Tensor1D<IN>>
    for Mlp<IN, INNER, OUT>
{
    type Output = Tensor1D<OUT>;

    fn forward(&self, input: Tensor1D<IN>) -> Self::Output {
        self.l2.forward(self.relu.forward(self.l1.forward(input)))
    }
}

// Impl module for batch forward pass
impl<const BATCH: usize, const IN: usize, const INNER: usize, const OUT: usize, T: Tape>
    Module<Tensor2D<BATCH, IN, T>> for Mlp<IN, INNER, OUT>
{
    type Output = Tensor2D<BATCH, OUT, T>;

    fn forward(&self, input: Tensor2D<BATCH, IN, T>) -> Self::Output {
        self.l2.forward(self.relu.forward(self.l1.forward(input)))
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
    let _y = model.forward(sample);

    // Forward pass with a batch of samples
    let batch: Tensor2D<BATCH_SIZE, 10> = Tensor2D::randn(&mut rng);
    let _y = model.forward(batch);
}
