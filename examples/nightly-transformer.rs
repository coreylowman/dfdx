//! Demonstrates how to use a transformer module on nightly rust.
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "nightly")]
fn main() {
    use dfdx::prelude::*;
    use rand::prelude::*;

    let mut rng = StdRng::seed_from_u64(0);
    let mut t: Transformer<16, 4, 3, 3, 8> = Default::default();
    t.reset_params(&mut rng);

    let src: Tensor3D<4, 12, 16> = TensorCreator::randn(&mut rng);
    let tgt: Tensor3D<4, 6, 16> = TensorCreator::randn(&mut rng);
    let _out: Tensor3D<4, 6, 16, _> = t.forward_mut((src.trace(), tgt));
}

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("Run with `cargo +nightly run ...` to run this example.");
}
