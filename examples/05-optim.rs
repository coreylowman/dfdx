//! Intro to dfdx::optim

use dfdx::{
    losses::mse_loss,
    nn::{BuildOnDevice, Linear, ModuleMut, ReLU, Tanh},
    optim::{Momentum, Optimizer, Sgd, SgdConfig},
    shapes::Rank2,
    tensor::{AsArray, Cpu, SampleTensor, Tensor},
    tensor_ops::Backward,
};

// first let's declare our neural network to optimze
type Mlp = (
    (Linear<5, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() {
    let dev: Cpu = Default::default();

    // The first step to optimizing is to initialize the optimizer.
    // Here we construct a stochastic gradient descent optimizer
    // for our Mlp.
    let mut sgd: Sgd<Mlp> = Sgd::new(SgdConfig {
        lr: 1e-1,
        momentum: Some(Momentum::Nesterov(0.9)),
        weight_decay: None,
    });

    // let's initialize our model and some dummy data
    let mut mlp = Mlp::build_on_device(&dev);
    let x: Tensor<Rank2<3, 5>, f32, _> = dev.sample_normal();
    let y: Tensor<Rank2<3, 2>, f32, _> = dev.sample_normal();

    // first we pass our gradient tracing input through the network.
    // since we are training, we use forward_mut() instead of forward
    let prediction = mlp.forward_mut(x.trace());

    // next compute the loss against the target dummy data
    let loss = mse_loss(prediction, y.clone());
    dbg!(loss.array());

    // extract the gradients
    let gradients = loss.backward();

    // the final step is to use our optimizer to update our model
    // given the gradients we've calculated.
    // This will modify our model!
    sgd.update(&mut mlp, gradients)
        .expect("Oops, there were some unused params");

    // let's do this a couple times to make sure the loss decreases!
    for i in 0..5 {
        let prediction = mlp.forward_mut(x.trace());
        let loss = mse_loss(prediction, y.clone());
        println!("Loss after update {i}: {:?}", loss.array());
        let gradients = loss.backward();
        sgd.update(&mut mlp, gradients)
            .expect("Oops, there were some unused params");
    }
}
