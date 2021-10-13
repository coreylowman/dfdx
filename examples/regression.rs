use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::Uniform,
};
use stag::{nn::{Chain, Linear, ReLU, Tanh}, optim::sgd::SgdConfig};
use stag::optim::sgd::Sgd;
use stag::prelude::*;
use std::time::Instant;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize target data
    let mut x: Tensor2D<64, 10> = Tensor2D::default();
    let mut y: Tensor2D<64, 2> = Tensor2D::default();
    x.randomize(&mut rng, &Uniform::new(-1.0, 1.0));
    y.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    // initialize optimizer & model
    // chain! expands to:
    // let module = <Linear<10, 32> as Default>::default()
    //     .chain::<ReLU>()
    //     .chain::<Linear<32, 32>>()
    //     .chain::<ReLU>()
    //     .chain::<Linear<32, 2>>()
    //     .chain::<Tanh>();
    let mut opt = Sgd::new(
        SgdConfig::default(),
        chain!(Linear<10, 32>, ReLU, Linear<32, 32>, ReLU, Linear<32, 2>, Tanh),
    );
    opt.init(&mut rng);

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // call forward & track derivaties
        let mut output = opt.forward_with_derivatives(&mut x);

        // compute loss
        let mut loss = (&mut output - &mut y).square().mean();

        // run backprop
        opt.step(&mut loss);

        println!("loss={:#.3} in {:?}", loss.data(), start.elapsed());
    }
}
