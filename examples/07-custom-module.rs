//! Demonstrates how to build a custom [nn::Module] without using tuples

use dfdx::{
    nn::modules::{Linear, Module, ModuleVisitor, ReLU, TensorCollection},
    prelude::BuildModule,
    shapes::{Dtype, Rank1, Rank2},
    tensor::{AutoDevice, SampleTensor, Tape, Tensor, Trace},
    tensor_ops::Device,
};

/// Custom model struct
/// This case is trivial and should be done with a tuple of linears and relus,
/// but it demonstrates how to build models with custom behavior
struct Mlp<const IN: usize, const INNER: usize, const OUT: usize, E: Dtype, D: Device<E>> {
    l1: Linear<IN, INNER, E, D>,
    l2: Linear<INNER, OUT, E, D>,
    relu: ReLU,
}

// TensorCollection lets you do several operations on Modules, including constructing them with
// randomized parameters, and iterating through or mutating all tensors in a model.
impl<const IN: usize, const INNER: usize, const OUT: usize, E: Dtype, D: Device<E>>
    TensorCollection<E, D> for Mlp<IN, INNER, OUT, E, D>
{
    // Type alias that specifies the how Mlp's type changes when using a different dtype and/or
    // device.
    type To<E2: Dtype, D2: Device<E2>> = Mlp<IN, INNER, OUT, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                // Define name of each field and how to access it, using ModuleField for Modules,
                // and TensorField for Tensors.
                Self::module("l1", |s| &s.l1, |s| &mut s.l1),
                Self::module("l2", |s| &s.l2, |s| &mut s.l2),
            ),
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |(l1, l2)| Mlp {
                l1,
                l2,
                relu: Default::default(),
            },
        )
    }
}

// impl Module for single item
impl<const IN: usize, const INNER: usize, const OUT: usize, E: Dtype, D: Device<E>>
    Module<Tensor<Rank1<IN>, E, D>> for Mlp<IN, INNER, OUT, E, D>
{
    type Output = Tensor<Rank1<OUT>, E, D>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<Rank1<IN>, E, D>) -> Result<Self::Output, D::Err> {
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
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
    > Module<Tensor<Rank2<BATCH, IN>, E, D, T>> for Mlp<IN, INNER, OUT, E, D>
{
    type Output = Tensor<Rank2<BATCH, OUT>, E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<Rank2<BATCH, IN>, E, D, T>) -> Result<Self::Output, D::Err> {
        let x = self.l1.try_forward(x)?;
        let x = self.relu.try_forward(x)?;
        self.l2.try_forward(x)
    }
}

fn main() {
    // Rng for generating model's params
    let dev = AutoDevice::default();

    // Construct model
    let model = Mlp::<10, 512, 20, f32, AutoDevice>::build(&dev);

    // Forward pass with a single sample
    let item: Tensor<Rank1<10>, f32, _> = dev.sample_normal();
    let _: Tensor<Rank1<20>, f32, _> = model.forward(item);

    // Forward pass with a batch of samples
    let batch: Tensor<Rank2<32, 10>, f32, _> = dev.sample_normal();
    let _: Tensor<Rank2<32, 20>, f32, _, _> = model.forward(batch.leaky_trace());
}
