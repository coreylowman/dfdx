//! So far we have only used single layers in the examples.
//! But real world models have many layers and sub models.
//! In this example we can see a simple way to compose many layers together
//! sequentially.

use dfdx::prelude::*;

/// Here we define a simple feedforward network with 3 layers.
/// the `#[derive(Sequential)]` means the built module will execute
/// the layers in **specification** order. So it will go:
/// 1. linear1
/// 2. act1
/// 3. linear2
/// 4. act2
/// 5. linear3
#[derive(Debug, Clone, Sequential)]
struct MlpConfig {
    // Linear with compile time input size & runtime known output size
    linear1: LinearConfig<Const<784>, usize>,
    act1: ReLU,
    // Linear with runtime input & output size
    linear2: LinearConfig<usize, usize>,
    act2: Tanh,
    // Linear with runtime input & compile time output size.
    linear3: LinearConfig<usize, Const<10>>,
}

fn main() {
    let dev = AutoDevice::default();

    // We can't use `Default::default()` anymore because we need the runtime values.
    let arch = MlpConfig {
        linear1: LinearConfig::new(Const, 512),
        act1: Default::default(),
        linear2: LinearConfig::new(512, 256),
        act2: Default::default(),
        linear3: LinearConfig::new(256, Const),
    };

    // Same way of building it.
    let m = dev.build_module::<f32>(arch);

    // The built module has fields that are named exactly the same as the config struct.
    dbg!(&m.linear1);
    dbg!(&m.act1);
    dbg!(&m.linear2);
    dbg!(&m.act2);
    dbg!(&m.linear3);

    // Calling Module::forward on it.
    let x: Tensor<Rank2<10, 784>, f32, _> = dev.sample_normal();
    let y = m.forward(x);
    assert_eq!(y.shape(), &(Const::<10>, Const::<10>));
}
