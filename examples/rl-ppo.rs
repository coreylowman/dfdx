//! Implements the reinforcement learning algorithm Proximal Policy Optimization (PPO) on random data.

use dfdx::{
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
};
use std::time::Instant;

const BATCH: usize = 64;
const STATE: usize = 4;
const ACTION: usize = 2;

type PolicyNetwork = (
    (Linear<STATE, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    Linear<32, ACTION>,
);

fn main() {
    let dev: Cpu = Default::default();

    let state = dev.randn::<Rank2<BATCH, STATE>>();
    let mut i = 0;
    let action: Tensor<Rank1<BATCH>, usize, _> = dev.tensor([(); BATCH].map(|_| {
        i += 1;
        i % ACTION
    }));
    let advantage = dev.randn::<Rank1<BATCH>>();

    // initiliaze model - all weights are 0s
    let mut pi_net: PolicyNetwork = dev.build();
    let target_pi_net: PolicyNetwork = pi_net.clone();

    let mut sgd = Sgd::new(SgdConfig {
        lr: 1e-1,
        momentum: Some(Momentum::Nesterov(0.9)),
        weight_decay: None,
    });

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // old_log_prob_a = log(P(action | state, target_pi_net))
        let old_logits = target_pi_net.forward(state.clone());
        let old_log_prob_a = old_logits.log_softmax::<Axis<1>>().select(action.clone());

        // log_prob_a = log(P(action | state, pi_net))
        let logits = pi_net.forward(state.trace());
        let log_prob_a = logits.log_softmax::<Axis<1>>().select(action.clone());

        // ratio = P(action | state, pi_net) / P(action | state, target_pi_net)
        // but compute in log space and then do .exp() to bring it back out of log space
        let ratio = (log_prob_a - old_log_prob_a).exp();

        // because we need to re-use `ratio` a 2nd time, we need to do some tape manipulation here.
        let surr1 = ratio.with_empty_tape() * advantage.clone();
        let surr2 = ratio.clamp(0.8, 1.2) * advantage.clone();

        let ppo_loss = -(surr2.minimum(surr1).mean());

        let loss_v = ppo_loss.array();

        // run backprop
        let gradients = ppo_loss.backward();

        // update weights with optimizer
        sgd.update(&mut pi_net, gradients).expect("Unused params");

        println!("loss={:#} in {:?}", loss_v, start.elapsed());
    }
}
