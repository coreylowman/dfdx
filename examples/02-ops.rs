//! Intro to dfdx::tensor_ops

use dfdx::{
    arrays::{Rank0, Rank1, Rank2},
    tensor::{AsArray, Cpu, RandnTensor},
    tensor_ops::{MeanTo, TryMatMul},
};

fn main() {
    let dev: Cpu = Default::default();

    let a = dev.randn::<Rank2<2, 3>>();
    dbg!(a.array());

    let b = dev.randn::<Rank2<2, 3>>();
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
    let f = e.mean_to::<Rank0>();
    dbg!(f.array());

    // and of course you can chain all of these together
    let _ = dev
        .randn::<Rank2<5, 10>>()
        .clamp(-1.0, 1.0)
        .exp()
        .abs()
        .powf(0.5)
        / 2.0;

    // then we have things like matrix and vector multiplication:
    let a = dev.randn::<Rank2<3, 5>>();
    let b = dev.randn::<Rank2<5, 7>>();
    let c = a.matmul(b);
    dbg!(c.array());

    // which even the outer product between two vectors!
    let a = dev.randn::<Rank1<3>>();
    let b = dev.randn::<Rank1<7>>();
    let c = a.matmul(b);
    dbg!(c.array());
}
