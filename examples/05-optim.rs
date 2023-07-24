//! Intro to dfdx::optim

use dfdx::{
    losses::mse_loss,
    nn::builders::*,
    optim::{Momentum, Optimizer, Sgd, SgdConfig},
    shapes::Rank2,
    tensor::{AsArray, AutoDevice, SampleTensor, Tensor, Trace},
    tensor_ops::Backward,
};

// first let's declare our neural network to optimze
type Mlp = (
    (Linear<5, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() {
    let dev = AutoDevice::default();

    // First randomly initialize our model
    let mut mlp = dev.build_module::<Mlp, f32>();

    // Then we allocate some gradients for it, so we don't re-allocate all the time
    let mut grads = mlp.alloc_grads();

    // Here we construct a stochastic gradient descent optimizer for our Mlp.
    let mut sgd = Sgd::new(
        &mlp,
        SgdConfig {
            lr: 1e-1,
            momentum: Some(Momentum::Nesterov(0.9)),
            weight_decay: None,
        },
    );

    // let's initialize some dummy data to optimize with
    let x: Tensor<Rank2<3, 5>, f32, _> = dev.sample_normal();
    let y: Tensor<Rank2<3, 2>, f32, _> = dev.sample_normal();

    // first we pass our gradient tracing input through the network.
    // since we are training, we use forward_mut() instead of forward
    let prediction = mlp.forward_mut(x.trace(grads));

    // next compute the loss against the target dummy data
    let loss = mse_loss(prediction, y.clone());
    dbg!(loss.array());

    // run backprop to extract the gradients
    grads = loss.backward();

    // the final step is to use our optimizer to update our model
    // given the gradients we've calculated.
    // This will modify our model!
    sgd.update(&mut mlp, &grads)
        .expect("Oops, there were some unused params");

    // now we also have to zero the gradients since we juts used them!
    mlp.zero_grads(&mut grads);

    // let's do this a couple times to make sure the loss decreases!
    for i in 0..5 {
        let prediction = mlp.forward_mut(x.trace(grads));
        let loss = mse_loss(prediction, y.clone());
        println!("Loss after update {i}: {:?}", loss.array());
        grads = loss.backward();
        sgd.update(&mut mlp, &grads)
            .expect("Oops, there were some unused params");
        mlp.zero_grads(&mut grads);
    }
}
