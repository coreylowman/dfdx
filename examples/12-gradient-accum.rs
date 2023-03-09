use dfdx::{gradients::Gradients, nn::traits::ZeroGrads, prelude::*};

#[cfg(not(feature = "cuda"))]
type Device = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
type Device = dfdx::tensor::Cuda;

fn main() {
    let dev: Device = Default::default();

    type Model = (Linear<2, 5>, ReLU, Linear<5, 10>, Tanh, Linear<10, 20>);
    let model = dev.build_module::<Model, f32>();

    let x: Tensor<Rank2<10, 2>, f32, _> = dev.sample_normal();

    // first we call .alloc_grads, which both pre-allocates gradients
    // and also marks non-parameter gradients as temporary.
    // this allows .backward() to drop temporary gradients.
    let mut grads: Gradients<f32, _> = model.alloc_grads();

    // using x.trace_into() instead of trace() allows us to
    // accumulate Gradients
    grads = model.forward(x.trace_into(grads)).mean().backward();

    // backward will return the same gradients object that we passed
    // into trace_into()
    grads = model.forward(x.trace_into(grads)).mean().backward();

    // you can do this as many times as you want!
    grads = model.forward(x.trace_into(grads)).mean().backward();

    // finally, we can use ZeroGrads to zero out the accumulated gradients
    model.zero_grads(&mut grads);
    assert_eq!(grads.get(&model.0.weight).array(), [[0.0; 2]; 5]);
}
