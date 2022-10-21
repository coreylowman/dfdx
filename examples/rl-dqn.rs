//! Implements Deep Q Learning on random data.

use dfdx::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

const STATE_SIZE: usize = 4;
const ACTION_SIZE: usize = 2;

// our simple 2 layer feedforward network with ReLU activations
type QNetwork = (
    (Linear<STATE_SIZE, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    Linear<32, ACTION_SIZE>,
);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let state: Tensor2D<64, STATE_SIZE> = Tensor2D::randn(&mut rng);
    let action: [usize; 64] = [(); 64].map(|_| rng.gen_range(0..ACTION_SIZE));
    let reward: Tensor1D<64> = Tensor1D::randn(&mut rng);
    let done: Tensor1D<64> = Tensor1D::zeros();
    let next_state: Tensor2D<64, STATE_SIZE> = Tensor2D::randn(&mut rng);

    // initiliaze model - all weights are 0s
    let mut q_net: QNetwork = Default::default();
    q_net.reset_params(&mut rng);

    let target_q_net: QNetwork = q_net.clone();

    let mut sgd = Sgd::new(SgdConfig {
        lr: 1e-1,
        momentum: Some(Momentum::Nesterov(0.9)),
        weight_decay: None,
    });

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // targ_q = R + discount * max(Q(S'))
        // curr_q = Q(S)[A]
        // loss = mse(curr_q, targ_q)
        let next_q_values: Tensor2D<64, ACTION_SIZE> = target_q_net.forward(next_state.clone());
        let max_next_q: Tensor1D<64> = next_q_values.max();
        let target_q = 0.99 * mul(max_next_q, &(1.0 - done.clone())) + &reward;

        // forward through model, computing gradients
        let q_values = q_net.forward(state.trace());
        let action_qs: Tensor1D<64, OwnedTape> = q_values.select(&action);

        let loss = mse_loss(action_qs, &target_q);
        let loss_v = *loss.data();

        // run backprop
        let gradients = loss.backward();

        // update weights with optimizer
        sgd.update(&mut q_net, gradients).expect("Unused params");

        println!("q loss={:#.3} in {:?}", loss_v, start.elapsed());
    }
}
