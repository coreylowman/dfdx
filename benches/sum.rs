use std::time::Instant;

use dfdx::prelude::*;

#[cfg(feature = "cuda")]
type Dev = Cuda;

#[cfg(not(feature = "cuda"))]
type Dev = Cpu;

type Dtype = f32;
type InputShape = Rank4<32, 64, 128, 256>;

fn main() {
    println!("Benchmarking `sum`");
    println!("Device {}", std::any::type_name::<Dev>());
    println!("Dtype {}", std::any::type_name::<Dtype>());
    println!("Input shape {}", std::any::type_name::<InputShape>());
    println!();

    let dev: Dev = Default::default();

    loop {
        let img: Tensor<InputShape, Dtype, _> = dev.sample_normal();
        let grads = Gradients::without_leafs();

        let start = Instant::now();
        let y = img.traced(grads).sum();
        let fwd_dur = start.elapsed();

        let start = Instant::now();
        let _ = y.backward();
        let bwd_dur = start.elapsed();
        println!("fwd={:?} bwd={:?}", fwd_dur, bwd_dur);
    }
}
