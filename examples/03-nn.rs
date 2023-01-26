//! Intro to dfdx::nn

use dfdx::{
    nn::{Linear, Module, ModuleBuilder, ModuleMut, ReLU, ResetParams},
    shapes::{Const, Rank1, Rank2},
    tensor::{AsArray, AutoDevice, SampleTensor, Tensor, ZerosTensor},
};

fn main() {
    let dev: AutoDevice = Default::default();

    // nn exposes many different neural network types, like the Linear layer!
    // you can use Build::build to construct an initialized model
    let mut m: Linear<4, 2> = dev.build_module();

    // Build::reset_params also allows you to re-randomize the weights
    m.reset_params();

    // Modules act on tensors using either:
    // 1. `Module::forward`, which does not mutate the module
    let _: Tensor<Rank1<2>> = m.forward(dev.zeros::<Rank1<4>>());

    // 2. `ModuleMut::forward_mut()`, which may mutate the module
    let _: Tensor<Rank1<2>> = m.forward_mut(dev.zeros::<Rank1<4>>());

    // most of them can also act on many different shapes of tensors
    // here we see that Linear can also accept batches of inputs
    // Note: the Rank2 with a batch size of 10 in the input
    //       AND the output
    let _: Tensor<Rank2<10, 2>> = m.forward(dev.zeros::<Rank2<10, 4>>());

    // Even dynamic size is supported;
    let batch_size = 3;
    let _: Tensor<(usize, Const<2>)> = m.forward(dev.zeros_like(&(batch_size, Const)));

    // you can also combine multiple modules with tuples
    let mlp: (Linear<4, 2>, ReLU, Linear<2, 1>) = dev.build_module();

    // and of course forward passes the input through each module sequentially:
    let x = dev.sample_normal::<Rank1<4>>();
    let a: Tensor<Rank1<1>> = mlp.forward(x.clone());
    let b = mlp.2.forward(mlp.1.forward(mlp.0.forward(x)));
    assert_eq!(a.array(), b.array());
}
