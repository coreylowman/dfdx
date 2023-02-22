//! Implements the reinforcement learning algorithm Proximal Policy Optimization (PPO) on random data.

use dfdx::{
    gradients::Tape,
    nn::{
        self,
        modules::{Linear, ReLU, Tanh},
        Module,
    },
    optim::{Adam, AdamConfig},
    prelude::{smooth_l1_loss, Optimizer, OwnedTape, SplitInto},
    tensor::{AsArray, Cpu, SampleTensor, SplitTape, Tensor0D, Tensor1D, TensorFrom},
    tensor_ops::{Backward, BroadcastTo, MeanTo},
};
use std::f32;
use std::time::Instant;

const STATE_SIZE: usize = 2;
const INNER_SIZE: usize = 3;
const ACTION_SIZE: usize = 1;

type Network<const IN: usize, const INNER: usize, const OUT: usize> = (
    (Linear<IN, INNER, f32, Cpu>, ReLU), // 0: l1
    SplitInto<(
        (Linear<INNER, OUT, f32, Cpu>, Tanh), // 1.0.0: mu
        (Linear<INNER, OUT, f32, Cpu>, ReLU), // 1.0.1: std // TODO: should this be SoftPlus?
        Linear<INNER, OUT, f32, Cpu>,         // 1.0.2: value
    )>,
);

// Hyperparameters stolen from https://github.com/seungeunrho/minimalRL/blob/master/ppo-continuous.py for now
const LEARNING_RATE: f32 = 0.0003;
const GAMMA: f32 = 0.9;
const LAMBDA: f32 = 0.9;
const EPS_CLIP: f32 = 0.2;

fn main() {
    let dev: Cpu = Default::default();

    // initiliaze model - all weights are 0s
    let mut net: Network<STATE_SIZE, INNER_SIZE, ACTION_SIZE> = nn::BuildModule::build(&dev);

    let mut optimizer = Adam::new(
        &net,
        AdamConfig {
            lr: LEARNING_RATE,
            betas: [0.9, 0.999],
            eps: 1e-08,
            weight_decay: None,
        },
    );

    let mut state: Tensor1D<STATE_SIZE> = init_state(&dev);

    // run through training data
    for _i_epoch in 0..1500 {
        let start = Instant::now();

        // <>
        let (old_log_prob, action) = {
            // TODO: Should we avoid calculating _v? _v is the one with the original tape if that matters?
            let (mu, std, _v) = net.forward(state.retaped::<OwnedTape<Cpu>>());

            let action: Tensor1D<ACTION_SIZE> = dev.sample_normal() * std.clone() + mu.clone();
            let log_prob = {
                let variance = std.clone().powi(2);
                -(action.clone() - mu).powi(2) / (variance * 2.0)
                    - std.ln()
                    - (2.0f32 * f32::consts::PI).sqrt().ln()
            };
            (log_prob /*(std, mu, action)*/, action)
        };

        let (new_state, reward, done) = step_simulation(&dev, state.clone(), &action);

        // <calc advantage>
        let td_target: Tensor1D<ACTION_SIZE> =
            net.1 .0 .2.forward(net.0.forward(new_state.clone())) * f32::from(done as u8) * GAMMA
                + reward;
        let delta: Tensor1D<ACTION_SIZE> =
            td_target.clone() - net.1 .0 .2.forward(net.0.forward(state.clone()));
        let delta = delta.array();

        let mut advantage = 0.0;
        let advantage: Tensor1D<ACTION_SIZE, _> = {
            let mut data = delta
                .into_iter()
                .rev()
                .map(|delta| {
                    advantage = GAMMA * LAMBDA * advantage + delta;
                    advantage
                })
                .collect::<Vec<f32>>();
            data.reverse();
            dev.tensor(TryInto::<[_; ACTION_SIZE]>::try_into(data.as_slice()).unwrap())
        };
        // </calc advantage>

        let (mu, std, value)/* (
            Tensor1D<ACTION_SIZE, OwnedTape<Cpu>>,
            Tensor1D<ACTION_SIZE, OwnedTape<Cpu>>,
            Tensor1D<ACTION_SIZE, OwnedTape<Cpu>>,
        )*/ = net.forward(state.trace());
        let log_prob: Tensor1D<ACTION_SIZE, OwnedTape<Cpu>> = {
            let variance = std.retaped::<OwnedTape<Cpu>>().powi(2);
            -(action.trace() - mu).powi(2) / (variance * 2.0) - std.ln() - (2.0f32 * f32::consts::PI).sqrt().ln()
        }/*log_prob(std, mu, action)*/;

        // ratio = P(action | state, pi_net) / P(action | state, target_pi_net)
        // but compute in log space and then do .exp() to bring it back out of log space
        let ratio = (log_prob - old_log_prob).exp();

        // because we need to re-use `ratio` a 2nd time, we need to do some tape manipulation here.
        let surr1: Tensor1D<ACTION_SIZE, OwnedTape<Cpu>> =
            ratio.with_empty_tape() * advantage.clone();
        let surr2: Tensor1D<ACTION_SIZE, OwnedTape<Cpu>> =
            ratio.clamp(1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advantage;

        let loss = -surr2.minimum(surr1) + smooth_l1_loss(value, td_target, 1.0).broadcast();
        let loss: Tensor0D<OwnedTape<Cpu>> = loss.mean();

        let loss_v = loss.array();

        // TODO: How to nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        // run backprop
        let gradients = loss.backward();

        // update weights with optimizer
        optimizer
            .update(&mut net, gradients)
            .expect("Failed to update");

        state = if done { init_state(&dev) } else { new_state };
        println!("loss={:#?} in {:?}", loss_v, start.elapsed());
    }
}

fn init_state(dev: &Cpu) -> Tensor1D<STATE_SIZE> {
    let initial_temp = rand::random::<f32>() * 40.0;
    let initial_heater_power = 0.1;

    println!("Initial state created:");
    println!("  temp: {initial_temp}");
    println!("  power: {initial_heater_power}");

    tensor_from_state(dev, initial_temp, initial_heater_power)
}

fn step_simulation<T: Tape<Cpu>>(
    dev: &Cpu,
    old_state: Tensor1D<STATE_SIZE, T>,
    action: &Tensor1D<ACTION_SIZE, T>,
) -> (Tensor1D<STATE_SIZE>, f32, bool) {
    let new_heater_pwer = action.array()[0];
    let new_state = {
        let (last_temp, last_heater_power) = tensor_to_state(&old_state);
        println!(
            "State: temp={}, power={}",
            last_temp + last_heater_power,
            new_heater_pwer
        );
        tensor_from_state(dev, last_temp + last_heater_power, new_heater_pwer)
    };
    let reward = calculate_reward(&new_state, action);
    let is_done = reward > 19.9 || new_heater_pwer > 1.0; // Don't allow too high heater power

    (new_state, reward, is_done)
}

fn calculate_reward<T: Tape<Cpu>>(
    state: &Tensor1D<STATE_SIZE>,
    action: &Tensor1D<ACTION_SIZE, T>,
) -> f32 {
    let new_heater_power = action.array()[0];
    let (temperature, _) = tensor_to_state(state);

    // Don't allow too high heater power
    if new_heater_power > 1.0 {
        return 0.0;
    }

    20.0 - (temperature - 20.0).abs() // Highest reward at 20 degrees, lower the further away
}

fn tensor_to_state<T: Tape<Cpu>>(state: &Tensor1D<STATE_SIZE, T>) -> (f32, f32) {
    let [temperature_factor, last_heater_power] = state.array();

    // Rescale temperature to (most often) be within 0..1
    (temperature_factor * 40.0, last_heater_power)
}

fn tensor_from_state(dev: &Cpu, temperature: f32, last_heater_power: f32) -> Tensor1D<STATE_SIZE> {
    dev.tensor([temperature / 40.0, last_heater_power])
}

/*
fn log_prob<S: Shape>(
    std: Tensor<S, f32, Cpu>,
    mu: Tensor<S, f32, Cpu>,
    value: Tensor<S, f32, Cpu>,
) -> Tensor<S, f32, Cpu> {
    let variance = std.powi(2);
    -(value - mu).powi(2) / (variance * 2.0) - std.ln() - (2.0f32 * f32::consts::PI).sqrt().ln()
}
*/
