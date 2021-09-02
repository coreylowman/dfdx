mod gradients;
mod nn;
mod optim;
mod tensor;
mod traits;

use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    gradients::GradientTape,
    nn::Linear,
    optim::sgd::Sgd,
    tensor::Tensor1D,
    traits::{Module, Optimizer, Params, Tensor},
};

#[derive(Default, Debug)]
struct MyCoolNN {
    l1: Linear<5, 4>,
    l2: Linear<4, 3>,
    l3: Linear<3, 2>,
}

impl Params for MyCoolNN {
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.l1.randomize(rng);
        self.l2.randomize(rng);
        self.l3.randomize(rng);
    }

    fn register(&mut self, tape: &mut GradientTape) {
        self.l1.register(tape);
        self.l2.register(tape);
        self.l3.register(tape);
    }

    fn update(&mut self, tape: &GradientTape) {
        self.l1.update(tape);
        self.l2.update(tape);
        self.l3.update(tape);
    }
}

impl Module for MyCoolNN {
    type Input = Tensor1D<5>;
    type Output = Tensor1D<2>;

    fn forward(&mut self, x0: &mut Self::Input) -> Self::Output {
        let mut x1 = self.l1.forward(x0);
        let mut x2 = self.l2.forward(&mut x1);
        let x3 = self.l3.forward(&mut x2);
        x3
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut opt: Sgd<MyCoolNN> = Default::default();
    opt.randomize(&mut rng);

    let mut x: Tensor1D<5> = Default::default();
    x.randomize(&mut rng);
    println!("x={:?}", x);

    let mut y: Tensor1D<2> = Default::default();
    y.randomize(&mut rng);
    println!("y={:?}", y);

    for _ in 0..15 {
        let mut output = opt.forward_with_derivatives(&mut x);

        let mut loss = (&mut output - &mut y).square().mean();
        println!(
            "y={:#} output={:#} loss={:#}",
            y.data(),
            output.data(),
            loss.data()
        );

        opt.step(&mut loss);
    }

    println!("{:#?}", opt.l1);
    println!("{:#?}", opt.l2);
    println!("{:#?}", opt.l3);
}
