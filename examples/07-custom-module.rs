// // TODO: fix this
// //! Demonstrates how to build a custom [nn::Module] without using tuples

// use dfdx::{
    // nn::modules::{Linear, Module, ModuleVisitor, ModuleVisitorOutput, ReLU, TensorCollection},
    // shapes::{Dtype, Rank1, Rank2},
    // tensor::{HasErr, SampleTensor, Tape, Tensor},
    // tensor_ops::Device,
// };

// #[cfg(not(feature = "cuda"))]
// type Dev = dfdx::tensor::Cpu;

// #[cfg(feature = "cuda")]
// type Dev = dfdx::tensor::Cuda;

// type Err = <Dev as HasErr>::Err;

// /// Custom model struct
// /// This case is trivial and should be done with a tuple of linears and relus,
// /// but it demonstrates how to build models with custom behavior
// struct Mlp<const IN: usize, const INNER: usize, const OUT: usize> {
    // l1: Linear<IN, INNER, f32, Dev>,
    // l2: Linear<INNER, OUT, f32, Dev>,
    // relu: ReLU,
// }

// // TensorCollection lets you do several operations on Modules, including constructing them with
// // randomized paramters
// impl<const IN: usize, const INNER: usize, const OUT: usize> TensorCollection<f32, Dev>
    // for Mlp<IN, INNER, OUT>
// {
    // type Output<E2: Dtype, D2: Device<E2>> = Mlp<IN, INNER, OUT>;

    // fn iter_tensors<E2: Dtype, D2: Device<E2>, V: ModuleVisitor<Self, f32, Dev, E2, D2>>(
        // visitor: &mut V,
    // ) -> ModuleVisitorOutput<V::Func, Self, f32, Dev, E2, D2> {
        // let l1 = visitor.visit_module("l1", |s| &s.l1, |s| &mut s.l1)?;
        // let l2 = visitor.visit_module("l2", |s| &s.l2, |s| &mut s.l2)?;

        // Ok(dfdx::try_some!(Mlp {
            // l1: l1?,
            // l2: l2?,
            // relu: Default::default(),
        // }))
    // }
// }

// // impl Module for single item
// impl<const IN: usize, const INNER: usize, const OUT: usize> Module<Tensor<Rank1<IN>, f32, Dev>>
    // for Mlp<IN, INNER, OUT>
// {
    // type Output = Tensor<Rank1<OUT>, f32, Dev>;
    // type Error = Err;

    // fn try_forward(&self, x: Tensor<Rank1<IN>, f32, Dev>) -> Result<Self::Output, Err> {
        // let x = self.l1.try_forward(x)?;
        // let x = self.relu.try_forward(x)?;
        // self.l2.try_forward(x)
    // }
// }

// // impl Module for batch of items
// impl<
        // const BATCH: usize,
        // const IN: usize,
        // const INNER: usize,
        // const OUT: usize,
        // T: Tape<f32, Dev>,
    // > Module<Tensor<Rank2<BATCH, IN>, f32, Dev, T>> for Mlp<IN, INNER, OUT>
// {
    // type Output = Tensor<Rank2<BATCH, OUT>, f32, Dev, T>;
    // type Error = Err;

    // fn try_forward(&self, x: Tensor<Rank2<BATCH, IN>, f32, Dev, T>) -> Result<Self::Output, Err> {
        // let x = self.l1.try_forward(x)?;
        // let x = self.relu.try_forward(x)?;
        // self.l2.try_forward(x)
    // }
// }

// fn main() {
    // // Rng for generating model's params
    // let dev = Device::default();

    // // Construct model
    // let model = Mlp::<10, 512, 20>::build(&dev);

    // // Forward pass with a single sample
    // let item: Tensor<Rank1<10>, f32, _> = dev.sample_normal();
    // let _: Tensor<Rank1<20>, f32, _> = model.forward(item);

    // // Forward pass with a batch of samples
    // let batch: Tensor<Rank2<32, 10>, f32, _> = dev.sample_normal();
    // let _: Tensor<Rank2<32, 20>, f32, _, _> = model.forward(batch.trace());
// }

fn main() {}
