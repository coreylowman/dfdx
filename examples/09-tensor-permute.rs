//! Demonstrates how to re-order (permute/transpose) the axes of a tensor

use dfdx::arrays::{Axes3, Rank3};
use dfdx::tensor::{Cpu, Tensor, ZerosTensor};
use dfdx::tensor_ops::{PermuteInto, PermuteTo};

fn main() {
    let dev: Cpu = Default::default();

    let a: Tensor<Rank3<3, 5, 7>, f32, _> = dev.zeros();

    // permuting is as easy as just expressing the desired shape
    // note that we are reversing the order of the axes here!
    let b = a.permute_to::<Rank3<7, 5, 3>>();

    // we can do any of the expected combinations!
    let _ = b.permute_to::<Rank3<5, 7, 3>>();

    // Just like broadcast/reduce there are times when
    // type inference is impossible because of ambiguities.
    // You can specify axes explicitly to get aroudn this.
    let c: Tensor<Rank3<1, 1, 1>, f32, _> = dev.zeros();
    let _ = PermuteInto::<_, Axes3<1, 0, 2>>::permute(c);
    // NOTE: fails with "Multiple impls satisfying..."
    // let _ = c.permute_to::<Rank3<1, 1, 1>>();
}
