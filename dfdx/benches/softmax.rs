use std::time::Instant;

use dfdx_core::prelude::*;

#[cfg(feature = "cuda")]
type Dev = Cuda;

#[cfg(not(feature = "cuda"))]
type Dev = Cpu;

type Dtype = f32;
type InputShape = Rank4<32, 64, 128, 256>;
type Ax = Axis<3>;

fn main() {
    println!("Benchmarking `softmax` {}", std::any::type_name::<Ax>());
    println!("Device {}", std::any::type_name::<Dev>());
    println!("Dtype {}", std::any::type_name::<Dtype>());
    println!("Input shape {}", std::any::type_name::<InputShape>());
    println!();

    let dev: Dev = Default::default();

    loop {
        let img: Tensor<InputShape, Dtype, _> = dev.sample_normal();

        let start = Instant::now();
        let _ = img.softmax::<Ax>();
        dev.synchronize();
        let infer_dur = start.elapsed();

        let img: Tensor<InputShape, Dtype, _> = dev.sample_normal();

        let start = Instant::now();
        let y = img.leaky_traced().softmax::<Ax>();
        dev.synchronize();
        let fwd_dur = start.elapsed();

        let start = Instant::now();
        let _ = y.sum().backward();
        dev.synchronize();
        let bwd_dur = start.elapsed();

        println!("infer={infer_dur:?}, fwd={fwd_dur:?} bwd={bwd_dur:?}");
    }
}
