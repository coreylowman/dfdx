//! Demonstrates how to use a transformer module on nightly rust.
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "nightly")]
fn main() {
    use dfdx::prelude::*;

    let dev: AutoDevice = Default::default();
    let t: Transformer<16, 4, 3, 3, 8> = dev.build_module();

    let src: Tensor<Rank3<4, 12, 16>> = dev.sample_normal();
    let tgt: Tensor<Rank3<4, 6, 16>> = dev.sample_normal();
    let _: Tensor<Rank3<4, 6, 16>, _, _, _> = t.forward((src.trace(), tgt));
}

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("Run with `cargo +nightly run ...` to run this example.");
}
