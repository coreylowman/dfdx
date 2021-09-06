#![feature(generic_associated_types)]

use ndarray_rand::{
    rand::{rngs::StdRng, Rng, SeedableRng},
    rand_distr::Standard,
};

use rad::{module_collection, nn::Linear, optim::sgd::Sgd, tensor::Tensor2D, traits::*};

#[derive(Default, Debug)]
struct MyCoolNN {
    l1: Linear<5, 4>,
    l2: Linear<4, 3>,
    l3: Linear<3, 2>,
}

module_collection!(MyCoolNN[l1 l2 l3]);

impl Module for MyCoolNN {
    type Input<const B: usize> = Tensor2D<B, 5>;
    type Output<const B: usize> = Tensor2D<B, 2>;

    fn forward<const B: usize>(&mut self, x0: &mut Self::Input<B>) -> Self::Output<B> {
        let mut x1 = self.l1.forward(x0).relu();
        let mut x2 = self.l2.forward(&mut x1).relu();
        let x3 = self.l3.forward(&mut x2);
        x3
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut opt: Sgd<MyCoolNN> = Default::default();
    opt.randomize(&mut rng, &Standard);

    let mut x: Tensor2D<10, 5> = Default::default();
    x.randomize(&mut rng, &Standard);
    println!("x={:#}", x.data());

    let mut y: Tensor2D<10, 2> = Default::default();
    y.randomize(&mut rng, &Standard);
    println!("y={:#}", y.data());

    for _ in 0..15 {
        let mut output = opt.forward_with_derivatives(&mut x);

        let mut loss = (&mut output - &mut y).square().mean();
        println!(
            "loss={:#}",
            // y.data(),
            // output.data(),
            loss.data()
        );

        opt.step(&mut loss);
    }

    // println!("{:#?}", opt.l1);
    // println!("{:#?}", opt.l2);
    // println!("{:#?}", opt.l3);
}
