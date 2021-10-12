use ndarray_rand::rand::prelude::*;
use stag::nn::{Linear, ModuleChain, ReLU, Tanh};
use stag::prelude::*;

/*
Syntactic Sugar for
type MyMLP = ModuleChain<
    Linear<10, 32>,
    ModuleChain<
        ReLU<Tensor1D<32>>,
        ModuleChain<
            Linear<32, 32>,
            ModuleChain<ReLU<Tensor1D<32>>, ModuleChain<Linear<32, 2>, Tanh<Tensor1D<2>>>>,
        >,
    >,
>;
*/
type MyMLP = chain_modules!(Linear<10, 32>, ReLU<Tensor1D<32>>, Linear<32, 32>, ReLU<Tensor1D<32>>, Linear<32, 2>, Tanh<Tensor1D<2>>);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize the MLP
    let mut model: MyMLP = Default::default();
    model.init(&mut rng);

    // create a 1x10 tensor with zeros
    let mut x: Tensor2D<1, 10> = Default::default();

    // forward through the model
    let y = model.forward(&mut x);

    println!("{:#}", y.data());
    // [[-0.8459329, 0.9993481]]
}
