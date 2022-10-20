//! Intro to dfdx::optim

use rand::prelude::*;

use dfdx::arrays::HasArrayData;
use dfdx::gradients::{Gradients, OwnedTape};
use dfdx::losses::mse_loss;
use dfdx::nn::{Linear, ModuleMut, ReLU, ResetParams, Tanh};
use dfdx::optim::{Momentum, Optimizer, Sgd, SgdConfig};
use dfdx::tensor::{Tensor2D, TensorCreator};

// first let's declare our neural network to optimze
type Mlp = (
    (Linear<5, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // The first step to optimizing is to initialize the optimizer.
    // Here we construct a stochastic gradient descent optimizer
    // for our Mlp.
    let mut sgd: Sgd<Mlp> = Sgd::new(SgdConfig {
        lr: 1e-1,
        momentum: Some(Momentum::Nesterov(0.9)),
    });

    // let's initialize our model and some dummy data
    let mut mlp: Mlp = Default::default();
    mlp.reset_params(&mut rng);
    let x: Tensor2D<3, 5> = TensorCreator::randn(&mut rng);
    let y: Tensor2D<3, 2> = TensorCreator::randn(&mut rng);

    // first we pass our gradient tracing input through the network.
    // since we are training, we use forward_mut() instead of forward
    let prediction: Tensor2D<3, 2, OwnedTape> = mlp.forward_mut(x.trace());

    // next compute the loss against the target dummy data
    let loss = mse_loss(prediction, y.clone());
    dbg!(loss.data());

    // extract the gradients
    let gradients: Gradients = loss.backward();

    // the final step is to use our optimizer to update our model
    // given the gradients we've calculated.
    // This will modify our model!
    sgd.update(&mut mlp, gradients)
        .expect("Oops, there were some unused params");

    // let's do this a couple times to make sure the loss decreases!
    for i in 0..5 {
        let prediction = mlp.forward_mut(x.trace());
        let loss = mse_loss(prediction, y.clone());
        println!("Loss after update {i}: {:?}", loss.data());
        let gradients: Gradients = loss.backward();
        sgd.update(&mut mlp, gradients)
            .expect("Oops, there were some unused params");
    }
}
