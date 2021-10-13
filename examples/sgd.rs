use ndarray_rand::rand::prelude::*;
use stag::nn::Linear;
use stag::optim::sgd::{Sgd, SgdConfig};
use stag::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize target data
    let mut x = Tensor2D::<64, 5>::rand(&mut rng);
    let mut y = Tensor2D::<64, 2>::rand(&mut rng);

    // initialize optimizer & model
    let mut opt = Sgd::new(SgdConfig { lr: 0.5 }, Linear::<5, 2>::default());
    opt.init(&mut rng);
    println!("{:?}", opt.module);

    // call forward & track derivaties
    let mut output = opt.forward_with_derivatives(&mut x);

    // compute loss
    let mut loss = (&mut output - &mut y).square().mean();
    println!("loss={:#}", loss.data());
    // loss=1.420053

    // run backprop
    opt.step(&mut loss);
    // opt.step(&mut loss);
    println!("{:?}", opt.module);

    let mut output = opt.forward(&mut x);
    println!(
        "loss after 1 sgd step={:#}",
        (&mut output - &mut y).square().mean().data()
    );
    // loss after 1 sgd step=0.41276962
}
