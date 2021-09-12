use ndarray_rand::{
    rand::{rngs::StdRng, Rng, SeedableRng},
    rand_distr::Uniform,
};
use stag::{
    chain_modules, module_collection,
    nn::{traits::Module, Linear, ModuleChain, ReLU, Sin},
    optim::{
        sgd::{Sgd, SgdConfig},
        traits::Optimizer,
    },
    tensor::{traits::*, Tensor1D, Tensor2D},
};
use std::time::Instant;

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

type MyNiceChain = chain_modules!(Linear<10, 32>, Linear<32, 32>, Linear<32, 2>, Sin<Tensor1D<2>>);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut opt: Sgd<MyNiceChain> = Sgd {
        cfg: SgdConfig { lr: 1e-3 },
        module: Default::default(),
    };
    let mut x: Tensor2D<64, 10> = Default::default();
    let mut y: Tensor2D<64, 2> = Default::default();

    opt.randomize(&mut rng, &Uniform::new(-1.0, 1.0));
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
