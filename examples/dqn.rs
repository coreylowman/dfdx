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
    let action: [usize; 64] = {
        let mut values = [0; 64];
        for i in 0..64 {
            values[i] = rng.gen_range(0..ACTION_SIZE);
        }
        values
    };
    let reward: Tensor1D<64> = Tensor1D::randn(&mut rng);
    let done: Tensor1D<64> = Tensor1D::zeros();
    let next_state: Tensor2D<64, STATE_SIZE> = Tensor2D::randn(&mut rng);

    // initiliaze model - all weights are 0s
    let mut q_net: QNetwork = Default::default();
    q_net.reset_params(&mut rng);

    let target_q_net: QNetwork = q_net.clone();

    let mut sgd = Sgd::new(1e-1, Some(Momentum::Nesterov(0.9)));

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // targ_q = R + discount * max(Q(S'))
        // curr_q = Q(S)[A]
        // loss = mse(curr_q, targ_q)
        let next_q_values = target_q_net.forward(next_state.clone());
        let max_next_q = next_q_values.max_last_dim();
        let target_q = &reward + 0.99 * mul(&(1.0 - done.clone()), max_next_q);

        // forward through model, computing gradients
        let q_values = q_net.forward(state.trace());

        let loss = mse_loss(q_values.gather_last_dim(&action), &target_q);
        let loss_v = *loss.data();

        // run backprop
        let gradients = loss.backward();

        // update weights with optimizer
        sgd.update(&mut q_net, gradients);

        println!("q loss={:#.3} in {:?}", loss_v, start.elapsed());
    }
}
