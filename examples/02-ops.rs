//! Intro to dfdx::tensor_ops

use dfdx::{
    shapes::{Rank0, Rank1, Rank2},
    tensor::{AsArray, Cpu, SampleTensor, Tensor},
    tensor_ops::{MeanTo, TryMatMul},
};

fn main() {
    let dev: Cpu = Default::default();

    let a = dev.sample_normal::<Rank2<2, 3>>();
    dbg!(a.array());

    let b = dev.sample_normal::<Rank2<2, 3>>();
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
    let f = e.mean::<Rank0, _>();
    dbg!(f.array());

    // and of course you can chain all of these together
    let _ = dev
        .sample_normal::<Rank2<5, 10>>()
        .clamp(-1.0, 1.0)
        .exp()
        .abs()
        .powf(0.5)
        / 2.0;

    // then we have things like matrix and vector multiplication:
    let a: Tensor<Rank2<3, 5>, f32, _> = dev.sample_normal();
    let b: Tensor<Rank2<5, 7>, f32, _> = dev.sample_normal();
    let c = a.matmul(b);
    dbg!(c.array());

    // which even the outer product between two vectors!
    let a: Tensor<Rank1<3>, f32, _> = dev.sample_normal();
    let b: Tensor<Rank1<7>, f32, _> = dev.sample_normal();
    let c = a.matmul(b);
    dbg!(c.array());
}
