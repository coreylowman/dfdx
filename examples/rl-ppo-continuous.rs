//! Implements the reinforcement learning algorithm Proximal Policy Optimization (PPO) on random data.

use dfdx::{
    gradients::Tape,
    nn::{
        self,
        modules::{Linear, ReLU, Tanh},
        BuildModule, Module,
    },
    optim::{Adam, AdamConfig},
    prelude::{smooth_l1_loss, GradientUpdate, Optimizer, OwnedTape, ParamUpdater, UnusedTensors},
    shapes::Shape,
    tensor::{
        AsArray, Cpu, HasErr, SampleTensor, SplitTape, Tensor, Tensor0D, Tensor1D, Tensor2D,
        TensorFrom,
    },
    tensor_ops::{Backward, MeanTo},
};
use std::f32;
use std::time::Instant;

const BATCH_SIZE: usize = 64;
const STATE_SIZE: usize = 4;
const INNER_SIZE: usize = 128;
const ACTION_SIZE: usize = 2;

/// Custom model struct
#[derive(Clone)]
struct Network<const IN: usize, const INNER: usize, const OUT: usize> {
    l1: (Linear<IN, INNER, f32, Cpu>, ReLU),
    mu: (Linear<INNER, OUT, f32, Cpu>, Tanh),
    std: (Linear<INNER, OUT, f32, Cpu>, ReLU), // TODO: should this be SoftPlus?
    value: Linear<INNER, OUT, f32, Cpu>,
}

// BuildModule lets you randomize a model's parameters
impl<const IN: usize, const INNER: usize, const OUT: usize> nn::BuildModule<Cpu, f32>
    for Network<IN, INNER, OUT>
{
    fn try_build(device: &Cpu) -> Result<Self, <Cpu as HasErr>::Err> {
        Ok(Self {
            l1: BuildModule::try_build(device)?,
            mu: BuildModule::try_build(device)?,
            std: BuildModule::try_build(device)?,
            value: BuildModule::try_build(device)?,
        })
    }
}

// GradientUpdate lets you update a model's parameters using gradients
impl<const IN: usize, const INNER: usize, const OUT: usize> GradientUpdate<Cpu, f32>
    for Network<IN, INNER, OUT>
{
    fn update<U>(
        &mut self,
        updater: &mut U,
        unused: &mut UnusedTensors,
    ) -> Result<(), <Cpu as HasErr>::Err>
    where
        U: ParamUpdater<Cpu, f32>,
    {
        self.l1.update(updater, unused)?;
        self.mu.update(updater, unused)?;
        self.std.update(updater, unused)?;
        self.value.update(updater, unused)?;
        Ok(())
    }
}

// impl Module for single item
impl<const IN: usize, const INNER: usize, const OUT: usize, T: Tape<Cpu>>
    nn::Module<Tensor1D<IN, T>> for Network<IN, INNER, OUT>
{
    type Output = (Tensor1D<OUT, T>, Tensor1D<OUT, T>, Tensor1D<OUT, T>);

    fn forward(&self, x: Tensor1D<IN, T>) -> Self::Output {
        let x = self.l1.forward(x);
        (
            self.mu.forward(x.retaped()),
            self.std.forward(x.retaped()),
            self.value.forward(x),
        )
    }
}

// impl Module for batch of items
impl<const BATCH: usize, const IN: usize, const INNER: usize, const OUT: usize, T: Tape<Cpu>>
    nn::Module<Tensor2D<BATCH, IN, T>> for Network<IN, INNER, OUT>
{
    type Output = (
        Tensor2D<BATCH, OUT, T>,
        Tensor2D<BATCH, OUT, T>,
        Tensor2D<BATCH, OUT, T>,
    );

    fn forward(&self, x: Tensor2D<BATCH, IN, T>) -> Self::Output {
        let x = self.l1.forward(x);
        (
            self.mu.forward(x.retaped()),
            self.std.forward(x.retaped()),
            self.value.forward(x),
        )
    }
}

// Hyperparameters stolen from https://github.com/seungeunrho/minimalRL/blob/master/ppo-continuous.py for now
const LEARNING_RATE: f32 = 0.0003;
const GAMMA: f32 = 0.9;
const LAMBDA: f32 = 0.9;
const EPS_CLIP: f32 = 0.2;
const K_EPOCH: usize = 10;
const ROLLOUT_LENGTH: usize = 3;
const BUFFER_SIZE: usize = 30;
const MINIBATCH_SIZE: usize = 32;

fn main() {
    let dev: Cpu = Default::default();

    // initiliaze model - all weights are 0s
    let mut net: Network<STATE_SIZE, INNER_SIZE, ACTION_SIZE> = nn::BuildModule::build(&dev);
    let target_net = net.clone();

    let mut optimizer = Adam::new(
        &net,
        AdamConfig {
            lr: LEARNING_RATE,
            betas: [0.9, 0.999],
            eps: 1e-08,
            weight_decay: None,
        },
    );

    let mut state = init_state();

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // <>
        let (old_log_prob, action) = {
            // TODO: Should we avoid calculating _v?
            let (mu, std, _v)= net.forward(state);
            let action: Tensor1D<ACTION_SIZE> = dev.sample_normal() * std + mu;
            let log_prob = {
                let variance = std.powi(2);
                -(action - mu).powi(2) / (variance * 2.0)
                    - std.ln()
                    - (2.0f32 * f32::consts::PI).sqrt().ln()
            };
            (log_prob /*(std, mu, action)*/, action)
        };

        let (new_state, reward, done) = step_simulation(state, &action);

        // <calc advantage>
        let td_target: Tensor1D<ACTION_SIZE> =
            net.value.forward(net.l1.forward(new_state)) * f32::from(done as u8) * GAMMA + reward;
        let delta = td_target.trace() - net.value.forward(net.l1.forward(state.trace()));
        let delta = delta.array();

        let mut advantage = 0.0;
        let advantage: Tensor1D<ACTION_SIZE, _> = dev.tensor(
            TryInto::<[_; ACTION_SIZE]>::try_into(delta
                .into_iter()
                .take(delta.len() - 1)
                .map(|delta| {
                    advantage = GAMMA * LAMBDA * advantage + delta;
                    advantage
                })
                .collect::<Vec<f32>>()
                .as_slice())
                .unwrap(),
        );
        // </calc advantage>

        let (mu, std, value): (Tensor1D<ACTION_SIZE, OwnedTape<Cpu>>, Tensor1D<ACTION_SIZE, OwnedTape<Cpu>>, Tensor1D<ACTION_SIZE, OwnedTape<Cpu>>) = net.forward(state.trace());
        let log_prob: Tensor1D<ACTION_SIZE, OwnedTape<Cpu>> = {
            let variance = std.powi(2);
            -(action.trace() - mu).powi(2) / (variance * 2.0) - std.ln() - (2.0f32 * f32::consts::PI).sqrt().ln()
        }/*log_prob(std, mu, action)*/;

        // ratio = P(action | state, pi_net) / P(action | state, target_pi_net)
        // but compute in log space and then do .exp() to bring it back out of log space
        let ratio = (log_prob - old_log_prob).exp();

        // because we need to re-use `ratio` a 2nd time, we need to do some tape manipulation here.
        let surr1: Tensor1D<ACTION_SIZE, OwnedTape<Cpu>> = ratio.with_empty_tape() * advantage.clone();
        let surr2: Tensor1D<ACTION_SIZE, OwnedTape<Cpu>> = ratio.clamp(1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advantage.clone();

        let loss = -surr2.minimum(surr1) + smooth_l1_loss(value, td_target, 1.0).array();
        let loss: Tensor0D<OwnedTape<Cpu>> = loss.mean();

        let loss_v = loss.array();

        // TODO: How to nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        // run backprop
        let gradients = loss.backward();

        // update weights with optimizer
        optimizer
            .update(&mut net, gradients)
            .expect("Failed to update");

        state = new_state;
        println!("loss={:#?} in {:?}", loss_v, start.elapsed());
    }
}

struct Transition {
    state: Tensor1D<STATE_SIZE>,
    last_action: Tensor1D<ACTION_SIZE>,
    reward: f32,
}

fn init_state() -> Tensor1D<STATE_SIZE> {
    todo!()
}

fn step_simulation<T: Tape<Cpu>>(
    old_state: Tensor1D<STATE_SIZE, T>,
    action: &Tensor1D<ACTION_SIZE, T>,
) -> (Tensor1D<STATE_SIZE>, f32, bool) {
    let new_state = todo!();
    let reward = calculate_reward(&new_state);
    let is_done = todo!();

    (new_state, reward, is_done)
}

fn calculate_reward(state: &Tensor1D<STATE_SIZE>) -> f32 {
    todo!()
}

fn log_prob<S: Shape>(
    std: Tensor<S, f32, Cpu>,
    mu: Tensor<S, f32, Cpu>,
    value: Tensor<S, f32, Cpu>,
) -> Tensor<S, f32, Cpu> {
    let variance = std.powi(2);
    -(value - mu).powi(2) / (variance * 2.0) - std.ln() - (2.0f32 * f32::consts::PI).sqrt().ln()
}
