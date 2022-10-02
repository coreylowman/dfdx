//! Implements the reinforcement learning algorithm Proximal Policy Optimization (PPO) on random data.

use dfdx::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

const STATE_SIZE: usize = 4;
const ACTION_SIZE: usize = 2;

type PolicyNetwork = (
    (Linear<STATE_SIZE, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    Linear<32, ACTION_SIZE>,
);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let state: Tensor2D<64, STATE_SIZE> = Tensor2D::randn(&mut rng);
    let action: [usize; 64] = [(); 64].map(|_| rng.gen_range(0..ACTION_SIZE));
    let advantage: Tensor1D<64> = Tensor1D::randn(&mut rng);

    // initiliaze model - all weights are 0s
    let mut pi_net: PolicyNetwork = Default::default();
    pi_net.reset_params(&mut rng);

    let target_pi_net: PolicyNetwork = pi_net.clone();

    let mut sgd = Sgd::new(SgdConfig {
        lr: 1e-1,
        momentum: Some(Momentum::Nesterov(0.9)),
    });

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // old_log_prob_a = log(P(action | state, target_pi_net))
        let old_logits = target_pi_net.forward(state.clone());
        let old_log_prob_a: Tensor1D<64> = old_logits.log_softmax::<Axis<1>>().select(&action);

        // log_prob_a = log(P(action | state, pi_net))
        let logits = pi_net.forward(state.trace());
        let log_prob_a: Tensor1D<64, OwnedTape> = logits.log_softmax::<Axis<1>>().select(&action);

        // ratio = P(action | state, pi_net) / P(action | state, target_pi_net)
        // but compute in log space and then do .exp() to bring it back out of log space
        let ratio = (log_prob_a - &old_log_prob_a).exp();

        // because we need to re-use `ratio` a 2nd time, we need to do some tape manipulation here.
        let r_ = ratio.duplicate();
        let (surr1, tape) = (ratio * &advantage).split_tape();
        let surr2 = (r_.put_tape(tape)).clamp(0.8, 1.2) * &advantage;

        let ppo_loss = -(minimum(surr2, &surr1).mean());

        let loss_v = *ppo_loss.data();

        // run backprop
        let gradients = ppo_loss.backward();

        // update weights with optimizer
        sgd.update(&mut pi_net, gradients).expect("Unused params");

        println!("loss={:#} in {:?}", loss_v, start.elapsed());
    }
}
