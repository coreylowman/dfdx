//! Intro to dfdx::tensor_ops

use dfdx::{
    shapes::{Const, Rank0, Rank1, Rank2},
    tensor::{AsArray, AutoDevice, SampleTensor, Tensor},
    tensor_ops::{MeanTo, RealizeTo, TryMatMul},
};

fn main() {
    let dev = AutoDevice::default();

    let a: Tensor<Rank2<2, 3>, f32, _> = dev.sample_normal();
    dbg!(a.array());

    // rust can infer the shape & dtype here because we add this
    // to a below!
    let b = dev.sample_normal();
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

    // binary and unary operations can also be performed on dynamically sized tensors
    let mut a: Tensor<(Const<3>, usize), f32, _> = dev.sample_uniform_like(&(Const, 5));
    a = a + 0.5;
    let b: Tensor<(usize, Const<5>), f32, _> = dev.sample_uniform_like(&(3, Const));
    // note the use of `realize`
    let _: Tensor<(Const<3>, usize), f32, _> = a + b.realize().expect("`b` should have 3 rows");

    // then we have things like matrix and vector multiplication:
    let a: Tensor<(usize, Const<5>), f32, _> = dev.sample_normal_like(&(3, Const));
    let b: Tensor<(usize, usize), f32, _> = dev.sample_normal_like(&(5, 7));
    // if type inference is not possible, we explicitly provide the shape for `realize`
    let _: Tensor<(usize, usize), f32, _> = a.matmul(
        b.realize::<(Const<5>, usize)>()
            .expect("`b` should have 5 rows"),
    );

    // which even the outer product between two vectors!
    let a: Tensor<Rank1<3>, f32, _> = dev.sample_normal();
    let b: Tensor<Rank1<7>, f32, _> = dev.sample_normal();
    let c = a.matmul(b);
    dbg!(c.array());

    // these operations are equal across devices
    #[cfg(feature = "cuda")]
    {
        use dfdx::{nn::ToDevice, tensor::Cpu};

        let cpu = Cpu::default();

        let a: Tensor<Rank1<3>, f32, _> = dev.sample_normal();
        let b: Tensor<Rank1<7>, f32, _> = dev.sample_normal();
        let a_cpu = a.to_device(&cpu);
        let b_cpu = b.to_device(&cpu);
        assert_eq!(a_cpu.matmul(b_cpu).array(), a.matmul(b).array());
    }
}
