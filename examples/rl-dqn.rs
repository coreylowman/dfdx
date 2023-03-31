//! Implements Deep Q Learning on random data.

use std::time::Instant;

use dfdx::{
    losses::huber_loss,
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
    tensor::AutoDevice,
};

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
    let dev = AutoDevice::default();
    // initialize model
    let mut q_net = dev.build_module::<QNetwork, f32>();
    let mut target_q_net = q_net.clone();

    let mut grads = q_net.alloc_grads();

    let mut sgd = Sgd::new(
        &q_net,
        SgdConfig {
            lr: 1e-1,
            momentum: Some(Momentum::Nesterov(0.9)),
            weight_decay: None,
        },
    );

    let state = dev.sample_normal::<Rank2<BATCH, STATE>>();
    let mut i = 0;
    let action: Tensor<Rank1<BATCH>, usize, _> = dev.tensor([(); BATCH].map(|_| {
        i += 1;
        i % ACTION
    }));
    let reward = dev.sample_normal::<Rank1<BATCH>>();
    let done = dev.zeros::<Rank1<BATCH>>();
    let next_state = dev.sample_normal::<Rank2<BATCH, STATE>>();

    // run through training data
    for epoch in 0..10 {
        let start = Instant::now();
        let mut total_loss = 0f32;

        for _step in 0..20 {
            // forward through model, computing gradients
            let q_values = q_net.forward(state.trace(grads));
            let action_qs = q_values.select(action.clone());

            // targ_q = R + discount * max(Q(S'))
            // curr_q = Q(S)[A]
            // loss = huber(curr_q, targ_q, 1)
            let next_q_values = target_q_net.forward(next_state.clone());
            let max_next_q = next_q_values.max::<Rank1<BATCH>, _>();
            let target_q = (max_next_q * (-done.clone() + 1.0)) * 0.99 + reward.clone();

            let loss = huber_loss(action_qs, target_q, 1.0);
            total_loss += loss.array();

            // run backprop
            grads = loss.backward();

            // update weights with optimizer
            sgd.update(&mut q_net, &grads).expect("Unused params");
            q_net.zero_grads(&mut grads);
        }
        target_q_net.clone_from(&q_net);

        println!(
            "Epoch {} in {:?}: q loss={:#.3}",
            epoch + 1,
            start.elapsed(),
            total_loss / 20.0
        );
    }
}
