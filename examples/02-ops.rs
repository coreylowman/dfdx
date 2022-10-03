//! Intro to dfdx::tensor_ops

use rand::prelude::*;

use dfdx::arrays::HasArrayData;
use dfdx::tensor::{Tensor0D, Tensor2D, TensorCreator};
use dfdx::tensor_ops::add;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let a: Tensor2D<2, 3> = TensorCreator::randn(&mut rng);
    dbg!(a.data());

    let b: Tensor2D<2, 3> = TensorCreator::randn(&mut rng);
    dbg!(b.data());

    // we can do binary operations like add two tensors together
    let c = add(a, &b);
    dbg!(c.data());

    // or unary operations like apply the `relu` function to each element
    let d = c.relu();
    dbg!(d.data());

    // we can add/sub/mul/div scalar values to tensors
    let e = d + 0.5;
    dbg!(e.data());

    // or reduce tensors to smaller sizes
    let f: Tensor0D = e.mean();
    dbg!(f.data());
}
