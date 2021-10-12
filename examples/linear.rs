use ndarray_rand::rand::prelude::*;
use stag::nn::Linear;
use stag::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize the MLP
    let mut model: Linear<4, 2> = Default::default();
    model.init(&mut rng);
    println!("{:?}", model);

    // create a 1x10 tensor with zeros
    let mut x: Tensor2D<1, 4> = Default::default();

    // forward through the model
    let y = model.forward(&mut x);

    println!("{:#}", y.data());
    // [[0.741256, -0.4756589]]
}
