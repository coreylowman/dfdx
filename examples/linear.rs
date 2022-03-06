use rand::prelude::*;
use rand_distr::StandardNormal;
use stag::nn::Linear;
use stag::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize the linear layer (ax + b)
    let mut model: Linear<4, 2> = Default::default();
    model.randomize(&mut rng, &StandardNormal);
    println!("{:?}", model);

    // create a 4 tensor with zeros
    let mut x: Tensor1D<4> = Default::default();

    // forward through the model
    let y = model.forward(&mut x);

    println!("{:#}", y.data());
    // [0.741256, -0.4756589]

    // create a batch of size 2 (2x4 tensor) with zeros
    let mut x: Tensor2D<2, 4> = Default::default();

    // forward through the model
    let y = model.forward(&mut x);

    println!("{:#}", y.data());
    // [[0.741256, -0.4756589],
    // [0.741256, -0.4756589]]
}
