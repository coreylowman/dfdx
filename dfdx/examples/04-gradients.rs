use dfdx_nn::{dfdx::prelude::*, BuildModuleExt, LinearConstConfig, Module, ReLU, Sequential};

#[derive(Clone, Default, Debug, Sequential)]
struct MlpConfig<const I: usize, const O: usize> {
    linear1: LinearConstConfig<I, 64>,
    act1: ReLU,
    linear2: LinearConstConfig<64, 64>,
    act2: ReLU,
    linear3: LinearConstConfig<64, O>,
}

fn main() {
    let dev: AutoDevice = Default::default();

    let arch: MlpConfig<784, 10> = Default::default();
    let mut module = dev.build_module::<f32>(arch);

    // We use `ZeroGrads::alloc_grads` to pre-allocate model gradients.
    // This will let us trace the model forward call without re-allocating gradients.
    use dfdx_nn::ZeroGrads;
    let mut grads = module.alloc_grads();

    // Here we accumulate gradients for 10 iterations.
    // Note that we pass in `grads` to `x.traced()`, and then receive the same object back
    // from `loss.backward()`.
    for _ in 0..10 {
        let x = dev.sample_normal::<Rank2<10, 784>>();
        let y = module.forward_mut(x.traced(grads));
        let loss = y.square().mean();
        grads = loss.backward();
    }
}
