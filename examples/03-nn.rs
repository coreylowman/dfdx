//! Intro to dfdx::nn

use dfdx::{
    nn::{BuildOnDevice, Linear, Module, ModuleMut, ReLU, ResetParams},
    shapes::{Const, Rank1, Rank2},
    tensor::{AsArray, Cpu, SampleTensor, Tensor, ZerosTensor},
};

fn main() {
    let dev: Cpu = Default::default();

    // nn exposes many different neural network types, like the Linear layer!
    // you can use Build::build to construct an initialized model
    let mut m = Linear::<4, 2>::build_on_device(&dev);

    // Build::reset_params also allows you to re-randomize the weights
    m.reset_params();

    // Modules act on tensors using either:
    // 1. `Module::forward`, which does not mutate the module
    let _: Tensor<Rank1<2>, f32, _> = m.forward(dev.zeros::<Rank1<4>>());

    // 2. `ModuleMut::forward_mut()`, which may mutate the module
    let _: Tensor<Rank1<2>, f32, _> = m.forward_mut(dev.zeros::<Rank1<4>>());

    // most of them can also act on many different shapes of tensors
    // here we see that Linear can also accept batches of inputs
    // Note: the Rank2 with a batch size of 10 in the input
    //       AND the output
    let _: Tensor<Rank2<10, 2>, f32, _> = m.forward(dev.zeros::<Rank2<10, 4>>());

    // Even dynamic size is supported;
    let batch_size = 3;
    let _: Tensor<(usize, Const<2>), f32, _> = m.forward(dev.zeros_like(&(batch_size, Const)));

    // you can also combine multiple modules with tuples
    type Mlp = (Linear<4, 2>, ReLU, Linear<2, 1>);
    let mlp = Mlp::build_on_device(&dev);

    // and of course forward passes the input through each module sequentially:
    let x: Tensor<Rank1<4>, f32, _> = dev.sample_normal();
    let a = mlp.forward(x.clone());
    let b = mlp.2.forward(mlp.1.forward(mlp.0.forward(x)));
    assert_eq!(a.array(), b.array());
}
