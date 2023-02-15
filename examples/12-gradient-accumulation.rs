use dfdx::{optim::Sgd, prelude::*};

fn main() {
    let dev: Cpu = Default::default();

    type Model = (Linear<2, 3>, ReLU, Linear<3, 4>, ReLU, Linear<4, 5>);
    let mut model = dev.build_module::<Model, f32>();

    let x: Tensor<Rank2<10, 2>, f32, _> = dev.sample_normal();
    let mut grads = model.forward(x.trace()).square().mean().backward();

    let x: Tensor<Rank2<10, 2>, f32, _> = dev.sample_normal();
    grads += model.forward(x.trace()).square().mean().backward();

    let x: Tensor<Rank2<10, 2>, f32, _> = dev.sample_normal();
    grads += model.forward(x.trace()).square().mean().backward();

    let mut opt = Sgd::new(&model, Default::default());
    opt.update(&mut model, grads).unwrap();
}
