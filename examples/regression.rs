use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Uniform;
use stag::prelude::*;
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
    let mut module: MLP = Default::default();
    module.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    let mut sgd = Sgd { lr: 1e-2 };

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        x.with_grad();
        let pred = module.forward(&x);
        let loss = sub(&pred, &y).square().mean();
        let gradients = sgd.compute_gradients(&loss);
        module.update_with_gradients(&gradients);

        println!("mse={:#.3} in {:?}", loss.data(), start.elapsed());
    }
}
