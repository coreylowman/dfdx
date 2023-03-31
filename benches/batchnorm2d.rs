use std::time::Instant;

use dfdx::prelude::*;

#[cfg(feature = "cuda")]
type Dev = Cuda;

#[cfg(not(feature = "cuda"))]
type Dev = Cpu;

type Model = BatchNorm2D<512>;
type Dtype = f32;
type InputShape = Rank4<64, 512, 28, 28>;

fn main() {
    println!("Benchmarking `BatchNorm2D`");
    println!("Device {}", std::any::type_name::<Dev>());
    println!("Dtype {}", std::any::type_name::<Dtype>());
    println!("Input shape {}", std::any::type_name::<InputShape>());
    println!();

    let dev: Dev = Default::default();
    let mut m = dev.build_module::<Model, Dtype>();

    loop {
        let img: Tensor<InputShape, Dtype, _> = dev.sample_normal();
        let grads = m.alloc_grads();

        let start = Instant::now();
        let out = m.forward_mut(img.traced(grads));
        let loss = out.square().mean();
        dev.synchronize();
        let fwd_dur = start.elapsed();

        let start = Instant::now();
        let _ = loss.backward();
        dev.synchronize();
        let bwd_dur = start.elapsed();
        println!("fwd={:?} bwd={:?}", fwd_dur, bwd_dur);
    }
}
