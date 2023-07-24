//! Implements the reinforcement learning algorithm Proximal Policy Optimization (PPO) on random data.

use std::time::Instant;

use dfdx::{
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
    tensor::AutoDevice,
};

const BATCH: usize = 64;
const STATE: usize = 4;
const ACTION: usize = 2;

type PolicyNetwork = (
    (Linear<STATE, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    Linear<32, ACTION>,
);

fn main() {
    let dev = AutoDevice::default();

    // initiliaze model - all weights are 0s
    let mut pi_net = dev.build_module::<PolicyNetwork, f32>();
    let mut target_pi_net = pi_net.clone();

    let mut grads = pi_net.alloc_grads();

    let mut sgd = Sgd::new(
        &pi_net,
        SgdConfig {
            lr: 1e-2,
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
    let advantage = dev.sample_normal::<Rank1<BATCH>>();

    // run through training data
    for epoch in 0..10 {
        let start = Instant::now();
        let mut total_loss = 0f32;

        for _step in 0..20 {
            // old_log_prob_a = log(P(action | state, target_pi_net))
            let old_logits = target_pi_net.forward(state.clone());
            let old_log_prob_a = old_logits.log_softmax::<Axis<1>>().select(action.clone());

            // log_prob_a = log(P(action | state, pi_net))
            let logits = pi_net.forward(state.trace(grads));
            let log_prob_a = logits.log_softmax::<Axis<1>>().select(action.clone());

            // ratio = P(action | state, pi_net) / P(action | state, target_pi_net)
            // but compute in log space and then do .exp() to bring it back out of log space
            let ratio = (log_prob_a - old_log_prob_a).exp();

            // because we need to re-use `ratio` a 2nd time, we need to do some tape manipulation here.
            let surr1 = ratio.with_empty_tape() * advantage.clone();
            let surr2 = ratio.clamp(0.8, 1.2) * advantage.clone();

            let ppo_loss = -(surr2.minimum(surr1).mean());

            total_loss += ppo_loss.array();

            // run backprop
            grads = ppo_loss.backward();

            // update weights with optimizer
            sgd.update(&mut pi_net, &grads).expect("Unused params");
            pi_net.zero_grads(&mut grads);
        }
        target_pi_net.clone_from(&pi_net);

        println!(
            "Epoch {} in {:?}: loss={:#.3}",
            epoch + 1,
            start.elapsed(),
            total_loss / 20.0
        );
    }
}
