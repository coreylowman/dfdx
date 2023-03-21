use dfdx::{
    nn::ZeroGrads,
    prelude::*,
    tensor::{AutoDevice, Gradients},
};

fn main() {
    let dev = AutoDevice::default();

    type Model = (Linear<2, 5>, ReLU, Linear<5, 10>, Tanh, Linear<10, 20>);
    let model = dev.build_module::<Model, f32>();

    let x: Tensor<Rank2<10, 2>, f32, _> = dev.sample_normal();

    // first we call .alloc_grads, which both pre-allocates gradients
    // and also marks non-parameter gradients as temporary.
    // this allows .backward() to drop temporary gradients.
    let mut grads: Gradients<f32, _> = model.alloc_grads();

    grads = model.forward(x.trace(grads)).mean().backward();

    // backward will return the same gradients object that we passed
    // into trace()
    grads = model.forward(x.trace(grads)).mean().backward();

    // you can do this as many times as you want!
    grads = model.forward(x.trace(grads)).mean().backward();

    // finally, we can use ZeroGrads to zero out the accumulated gradients
    model.zero_grads(&mut grads);
    assert_eq!(grads.get(&model.0.weight).array(), [[0.0; 2]; 5]);
}
