//! Demonstrates how to build a custom [nn::Module] without using tuples

use dfdx::{
    gradients::Tape,
    nn::{
        modules::{Linear, ReLU},
        BuildModule, Module,
    },
    shapes::{Rank1, Rank2},
    tensor::{HasErr, SampleTensor, Tensor},
};

#[cfg(not(feature = "cuda"))]
type Device = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
type Device = dfdx::tensor::Cuda;

type Err = <Device as HasErr>::Err;

/// Custom model struct
/// This case is trivial and should be done with a tuple of linears and relus,
/// but it demonstrates how to build models with custom behavior
struct Mlp<const IN: usize, const INNER: usize, const OUT: usize> {
    l1: Linear<IN, INNER, f32, Device>,
    l2: Linear<INNER, OUT, f32, Device>,
    relu: ReLU,
}

// BuildModule lets you randomize a model's parameters
impl<const IN: usize, const INNER: usize, const OUT: usize> BuildModule<Device, f32>
    for Mlp<IN, INNER, OUT>
{
    fn try_build(device: &Device) -> Result<Self, Err> {
        Ok(Self {
            l1: BuildModule::try_build(device)?,
            l2: BuildModule::try_build(device)?,
            relu: ReLU,
        })
    }
}

// impl Module for single item
impl<const IN: usize, const INNER: usize, const OUT: usize> Module<Tensor<Rank1<IN>, f32, Device>>
    for Mlp<IN, INNER, OUT>
{
    type Output = Tensor<Rank1<OUT>, f32, Device>;
    type Error = Err;

    fn try_forward(&self, x: Tensor<Rank1<IN>, f32, Device>) -> Result<Self::Output, Err> {
        let x = self.l1.try_forward(x)?;
        let x = self.relu.try_forward(x)?;
        self.l2.try_forward(x)
    }
}

// impl Module for batch of items
impl<
        const BATCH: usize,
        const IN: usize,
        const INNER: usize,
        const OUT: usize,
        T: Tape<f32, Device>,
    > Module<Tensor<Rank2<BATCH, IN>, f32, Device, T>> for Mlp<IN, INNER, OUT>
{
    type Output = Tensor<Rank2<BATCH, OUT>, f32, Device, T>;
    type Error = Err;

    fn try_forward(
        &self,
        x: Tensor<Rank2<BATCH, IN>, f32, Device, T>,
    ) -> Result<Self::Output, Err> {
        let x = self.l1.try_forward(x)?;
        let x = self.relu.try_forward(x)?;
        self.l2.try_forward(x)
    }
}

fn main() {
    // Rng for generating model's params
    let dev = Device::default();

    // Construct model
    let model = Mlp::<10, 512, 20>::build(&dev);

    // Forward pass with a single sample
    let item: Tensor<Rank1<10>, f32, _> = dev.sample_normal();
    let _: Tensor<Rank1<20>, f32, _> = model.forward(item);

    // Forward pass with a batch of samples
    let batch: Tensor<Rank2<32, 10>, f32, _> = dev.sample_normal();
    let _: Tensor<Rank2<32, 20>, f32, _, _> = model.forward(batch.trace());
}
