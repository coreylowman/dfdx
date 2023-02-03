//! Implements Deep Q Learning on random data.

use dfdx::{
    losses::mse_loss,
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
};
use std::time::Instant;

const BATCH: usize = 64;
const STATE: usize = 4;
const ACTION: usize = 2;

// our simple 2 layer feedforward network with ReLU activations
type QNetwork = (
    (Linear<STATE, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    Linear<32, ACTION>,
);

fn main() {
    let dev: Cpu = Default::default();

    let state = dev.sample_normal::<Rank2<BATCH, STATE>>();
    let mut i = 0;
    let action: Tensor<Rank1<BATCH>, usize, _> = dev.tensor([(); BATCH].map(|_| {
        i += 1;
        i % ACTION
    }));
    let reward = dev.sample_normal::<Rank1<BATCH>>();
    let done = dev.zeros::<Rank1<BATCH>>();
    let next_state = dev.sample_normal::<Rank2<BATCH, STATE>>();

    // initiliaze model
    let mut q_net = QNetwork::build(&dev);
    let target_q_net = q_net.clone();

    let mut sgd = Sgd::new(
        &q_net,
        SgdConfig {
            lr: 1e-1,
            momentum: Some(Momentum::Nesterov(0.9)),
            weight_decay: None,
        },
    );

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // targ_q = R + discount * max(Q(S'))
        // curr_q = Q(S)[A]
        // loss = mse(curr_q, targ_q)
        let next_q_values = target_q_net.forward(next_state.clone());
        let max_next_q = next_q_values.max::<Rank1<BATCH>, _>();

        let target_q = (max_next_q * (-done.clone() + 1.0)) * 0.99 + reward.clone();

        // forward through model, computing gradients
        let q_values = q_net.forward(state.trace());
        let action_qs = q_values.select(action.clone());

        let loss = mse_loss(action_qs, target_q);
        let loss_v = loss.array();

        // run backprop
        let gradients = loss.backward();

        // update weights with optimizer
        sgd.update(&mut q_net, gradients).expect("Unused params");

        println!("q loss={:#.3} in {:?}", loss_v, start.elapsed());
    }
}
