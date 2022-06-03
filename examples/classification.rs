use dfdx::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Uniform;
use std::time::Instant;

type MLP = (
    (Linear<10, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    Linear<32, 2>,
);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize target data
    let x: Tensor2D<64, 10> = Tensor2D::randn(&mut rng);
    let y: Tensor2D<64, 2> = Tensor2D::randn(&mut rng).softmax();

    // initialize model - all weights are 0s
    let mut mlp: MLP = Default::default();

    // randomize model weights
    mlp.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    // initialize our optimizer
    let mut sgd = Sgd::new(1e-2, None);

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // forward through model, computing gradients
        let pred = mlp.forward(x.trace());

        // compute loss
        let loss = cross_entropy_with_logits_loss(pred, &y);
        let loss_v /*: f32 */ = *loss.data();

        // run backprop
        let gradients = loss.backward();

        // update weights with optimizer
        sgd.update(&mut mlp, gradients);

        println!("cross entropy={:#.3} in {:?}", loss_v, start.elapsed());
    }

    mlp.save("classification.npz")
        .expect("failed to save model");
}
