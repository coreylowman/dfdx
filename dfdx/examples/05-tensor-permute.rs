//! Demonstrates how to re-order (permute/transpose) the axes of a tensor

use dfdx::{
    shapes::{Axes3, Rank3},
    tensor::{AutoDevice, Tensor, ZerosTensor},
    tensor_ops::PermuteTo,
};

fn main() {
    let dev = AutoDevice::default();

    let a: Tensor<Rank3<3, 5, 7>, f32, _> = dev.zeros();

    // permuting is as easy as just expressing the desired shape
    // note that we are reversing the order of the axes here!
    let b = a.permute::<Rank3<7, 5, 3>, _>();

    // we can do any of the expected combinations!
    let _ = b.permute::<Rank3<5, 7, 3>, _>();

    // Just like broadcast/reduce there are times when
    // type inference is impossible because of ambiguities.
    // You can specify axes explicitly to get aroudn this.
    let c: Tensor<Rank3<1, 1, 1>, f32, _> = dev.zeros();
    let _ = c.permute::<_, Axes3<1, 0, 2>>();
    // NOTE: fails with "Multiple impls satisfying..."
    // let _ = c.permute::<Rank3<1, 1, 1>, _>();
}
