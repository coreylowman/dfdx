use ndarray_rand::{
    rand::{rngs::StdRng, Rng, SeedableRng},
    rand_distr::Standard,
};

use rad::{
    chain_modules, module_collection,
    nn::{Linear, ModuleChain, ReLU},
    optim::sgd::Sgd,
    tensor::{Tensor1D, Tensor2D},
    traits::*,
};

#[derive(Default, Debug)]
struct MyCoolNN {
    l1: Linear<5, 4>,
    l2: Linear<4, 3>,
    l3: Linear<3, 2>,
}

module_collection!(MyCoolNN, [l1, l2, l3,]);

impl Module for MyCoolNN {
    type Input = Tensor1D<5>;
    type Output = Tensor1D<2>;

    fn forward<const B: usize>(
        &mut self,
        x0: &mut <Self::Input as Batch>::Batched<B>,
    ) -> <Self::Output as Batch>::Batched<B> {
        let mut x1 = self.l1.forward(x0).relu();
        let mut x2 = self.l2.forward(&mut x1).relu();
        let x3 = self.l3.forward(&mut x2);
        x3
    }
}

type LinearWithReLU<const I: usize, const O: usize> = ModuleChain<Linear<I, O>, ReLU<Tensor1D<O>>>;
type MyCoolChain =
    ModuleChain<LinearWithReLU<5, 4>, ModuleChain<LinearWithReLU<4, 3>, Linear<3, 2>>>;

type MyNiceChain = chain_modules!(LinearWithReLU<5, 4>, LinearWithReLU<4, 3>, Linear<3, 2>, );

// type InvalidChain = ModuleChain<Linear<5, 4>, Linear<3, 2>>;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut opt: Sgd<MyNiceChain> = Default::default();
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
