use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::Uniform,
};
use stag::nn::{Linear, ModuleChain, ReLU, Sin};
use stag::optim::sgd::Sgd;
use stag::prelude::*;
use std::time::Instant;

type MyMLP = chain_modules!(Linear<10, 32>, ReLU<Tensor1D<32>>, Linear<32, 32>, ReLU<Tensor1D<32>>, Linear<32, 2>, Sin<Tensor1D<2>>);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut opt: Sgd<MyMLP> = Default::default();
    opt.init(&mut rng);

    let mut x: Tensor2D<64, 10> = Default::default();
    let mut y: Tensor2D<64, 2> = Default::default();
    x.randomize(&mut rng, &Uniform::new(-1.0, 1.0));
    y.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    for _i_epoch in 0..15 {
        let start = Instant::now();

        let mut output = opt.forward_with_derivatives(&mut x);
        let mut loss = (&mut output - &mut y).square().mean();
        opt.step(&mut loss);

        println!("loss={:#.3} in {:?}", loss.data(), start.elapsed());
    }
}
