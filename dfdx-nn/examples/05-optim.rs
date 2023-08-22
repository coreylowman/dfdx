//! Intro to dfdx_nn::optim

use dfdx::prelude::*;
use dfdx_nn::{BuildModuleExt, LinearConstConfig, Module, ReLU, Sequential, Tanh, ZeroGrads};

// first let's declare our neural network to optimze
#[derive(Default, Clone, Sequential)]
#[built(Mlp)]
struct MlpConfig {
    l1: LinearConstConfig<5, 32>,
    act1: ReLU,
    l2: LinearConstConfig<32, 32>,
    act2: ReLU,
    l3: LinearConstConfig<32, 2>,
    act3: Tanh,
}

fn main() {
    let dev = AutoDevice::default();
    let mut mlp = dev.build_module_ext::<f32>(MlpConfig::default());
    let mut grads = mlp.alloc_grads();

    // let's initialize some dummy data to optimize with
    let x: Tensor<Rank2<3, 5>, f32, _> = dev.sample_normal();
    let y: Tensor<Rank2<3, 2>, f32, _> = dev.sample_normal();

    // To optimize a network we need an optimizer object. There are a few options,
    // here we will use Sgd.
    let mut sgd = dfdx_nn::Sgd::new(
        &mlp,
        dfdx_nn::SgdConfig {
            lr: 1e-1,
            momentum: Some(dfdx_nn::Momentum::Nesterov(0.9)),
            weight_decay: None,
        },
    );

    for i in 0..5 {
        // here we collect the gradients of the network as shown in 04-gradients
        let prediction = mlp.forward_mut(x.trace(grads));
        let loss = mse_loss(prediction, y.clone());
        println!("Loss after update {i}: {:?}", loss.array());
        grads = loss.backward();

        // the difference is we now call Optimizer::update with the gradients.
        use dfdx_nn::Optimizer;
        sgd.update(&mut mlp, &grads)
            .expect("Oops, there were some unused params");
        // and we also need to zero the gradients using `ZeroGrads::zero_grads`
        mlp.zero_grads(&mut grads);
    }
}
