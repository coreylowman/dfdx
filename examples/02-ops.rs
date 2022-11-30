//! Intro to dfdx::tensor_ops

use dfdx::tensor::storage::{AsArray, Cpu, Randn};
use dfdx::tensor::{Tensor0D, Tensor2D};
use dfdx::tensor_ops::{MeanTo, TryReLU};

fn main() {
    let dev: Cpu = Default::default();

    let a: Tensor2D<2, 3, _> = dev.randn();
    dbg!(a.array());

    let b: Tensor2D<2, 3, _> = dev.randn();
    dbg!(b.array());

    // we can do binary operations like add two tensors together
    let c = a + b;
    dbg!(c.array());

    // or unary operations like apply the `relu` function to each element
    let d = c.relu();
    dbg!(d.array());

    // we can add/sub/mul/div scalar values to tensors
    let e = d + 0.5;
    dbg!(e.array());

    // or reduce tensors to smaller sizes
    let f: Tensor0D<Cpu> = e.mean();
    dbg!(f.array());
}
