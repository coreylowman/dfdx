//! Implements the reinforcement learning algorithm Proximal Policy Optimization (PPO) on random data.

use dfdx::{
    gradients::Tape,
    nn::{
        self,
        modules::{Linear, ReLU, Tanh},
        BuildModule, Module,
    },
    optim::{Adam, AdamConfig},
    prelude::{smooth_l1_loss, GradientUpdate, ParamUpdater, UnusedTensors},
    tensor::{AsArray, Cpu, HasErr, SplitTape, Tensor1D, Tensor2D},
    tensor_ops::MeanTo,
};
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
impl<const IN: usize, const INNER: usize, const OUT: usize> nn::Module<Tensor1D<IN>>
    for Network<IN, INNER, OUT>
{
    type Output = (Tensor1D<OUT>, Tensor1D<OUT>, Tensor1D<OUT>);

    fn forward(&self, x: Tensor1D<IN>) -> Self::Output {
        let x = self.l1.forward(x);
        (
            self.mu.forward(x),
            self.std.forward(x),
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
            self.mu.forward(x),
            self.std.forward(x),
            self.value.forward(x),
        )
    }
}

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

    let mut state: Tensor1D<STATE_SIZE> = init_state();

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        // <calc advantage>
        let (new_state, reward, done) = step_simulation(action);

        let td_target = reward + GAMMA * net.value.forward(net.l1.forward(new_state)) * done;
        let delta = td_taget - net.value.forward(net.l1.forward(state));
        let delta = delta.array();

        let mut advantage = 0.0;
        let advantage = delta.take(delta.len() - 1).map(|delta| {
            advantage = GAMMA * LAMBDA * advantage + delta[0];
            advantage
        });
        // </calc advantage>

        let (mu, std, v) = net.forward(state);
        //let dist = mu.array().into_iter().zip(std.array()).map(|(mu, std)| Normal::new(mu, std)).collect(); // TODO: is this correct? How should this be done in a more "Tensor"y way?

        let log_prob: Tensor1D<INNER_SIZE> = todo!();

        // ratio = P(action | state, pi_net) / P(action | state, target_pi_net)
        // but compute in log space and then do .exp() to bring it back out of log space
        let ratio = (log_prob - old_log_prob).exp();

        // because we need to re-use `ratio` a 2nd time, we need to do some tape manipulation here.
        let surr1 = ratio.with_empty_tape() * advantage.clone();
        let surr2 = ratio.clamp(1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advantage.clone();

        let loss = -surr2.minimum(surr1) + smooth_l1_loss(v, td_target, 1.0);
        let loss = loss.mean();

        let loss_v = loss.array();

        // TODO: How to nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        // run backprop
        let gradients = loss.backward();

        // update weights with optimizer
        optimizer
            .update(&mut net, gradients)
            .expect("Unused params");

        println!("loss={:#} in {:?}", loss_v, start.elapsed());
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

fn step_simulation(action: &Tensor1D<ACTION_SIZE>) -> (Tensor1D<STATE_SIZE>, f32, bool) {
    todo!()
}
