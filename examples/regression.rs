use dfdx::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Uniform;
use std::time::Instant;

type MLP = (
    (Linear<10, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize target data
    let x: Tensor2D<64, 10> = Tensor2D::randn(&mut rng);
    let y: Tensor2D<64, 2> = Tensor2D::randn(&mut rng);

    // initialize optimizer & model
    let mut mlp: MLP = Default::default();
    mlp.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    let mut sgd = Sgd::new(1e-2, None);

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        let x = x.trace();
        let pred = mlp.forward(x);
        let loss = mse_loss(pred, &y);
        let loss_v = *loss.data();
        let gradients = loss.backward();
        sgd.update(&mut mlp, gradients);

        println!("mse={:#.3} in {:?}", loss_v, start.elapsed());
    }
}
