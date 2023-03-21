//! Demonstrates how to use a transformer module on nightly rust.

fn main() {
    use dfdx::{prelude::*, tensor::AutoDevice};

    let dev = AutoDevice::default();
    type Model = Transformer<16, 4, 3, 3, 8>;
    let t = dev.build_module::<Model, f32>();
    let grads = t.alloc_grads();

    let src: Tensor<Rank3<4, 12, 16>, f32, _> = dev.sample_normal();
    let tgt: Tensor<Rank3<4, 6, 16>, f32, _> = dev.sample_normal();
    let _: Tensor<Rank3<4, 6, 16>, _, _, _> = t.forward((src.trace(grads), tgt));
}
