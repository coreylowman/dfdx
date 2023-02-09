//! Demonstrates how to build a custom [nn::Module] without using tuples

use dfdx::{
    gradients::Tape,
    nn::{
        self,
        modules::{Linear, ReLU},
        BuildModule, Module,
    },
    shapes::{Rank1, Rank2},
    tensor::{Cpu, HasErr, SampleTensor, Tensor},
};

/// Custom model struct
/// This case is trivial and should be done with a tuple of linears and relus,
/// but it demonstrates how to build models with custom behavior
struct Mlp<const IN: usize, const INNER: usize, const OUT: usize> {
    l1: Linear<IN, INNER, f32, Cpu>,
    l2: Linear<INNER, OUT, f32, Cpu>,
    relu: ReLU,
}

// BuildModule lets you randomize a model's parameters
impl<const IN: usize, const INNER: usize, const OUT: usize> nn::BuildModule<Cpu, f32>
    for Mlp<IN, INNER, OUT>
{
    fn try_build(device: &Cpu) -> Result<Self, <Cpu as HasErr>::Err> {
        Ok(Self {
            l1: BuildModule::try_build(device)?,
            l2: BuildModule::try_build(device)?,
            relu: ReLU,
        })
    }
}

// impl Module for single item
impl<const IN: usize, const INNER: usize, const OUT: usize> nn::Module<Tensor<Rank1<IN>, f32, Cpu>>
    for Mlp<IN, INNER, OUT>
{
    type Output = Tensor<Rank1<OUT>, f32, Cpu>;

    fn forward(&self, x: Tensor<Rank1<IN>, f32, Cpu>) -> Self::Output {
        let x = self.l1.forward(x);
        let x = self.relu.forward(x);
        self.l2.forward(x)
    }
}

// impl Module for batch of items
impl<const BATCH: usize, const IN: usize, const INNER: usize, const OUT: usize, T: Tape<Cpu>>
    nn::Module<Tensor<Rank2<BATCH, IN>, f32, Cpu, T>> for Mlp<IN, INNER, OUT>
{
    type Output = Tensor<Rank2<BATCH, OUT>, f32, Cpu, T>;

    fn forward(&self, x: Tensor<Rank2<BATCH, IN>, f32, Cpu, T>) -> Self::Output {
        let x = self.l1.forward(x);
        let x = self.relu.forward(x);
        self.l2.forward(x)
    }
}

fn main() {
    // Rng for generating model's params
    let dev: Cpu = Default::default();

    // Construct model
    let model = Mlp::<10, 512, 20>::build(&dev);

    // Forward pass with a single sample
    let item: Tensor<Rank1<10>, f32, _> = dev.sample_normal();
    let _: Tensor<Rank1<20>, f32, Cpu> = model.forward(item);

    // Forward pass with a batch of samples
    let batch: Tensor<Rank2<32, 10>, f32, _> = dev.sample_normal();
    let _: Tensor<Rank2<32, 20>, f32, Cpu, _> = model.forward(batch.trace());
}
