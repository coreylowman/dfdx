use std::time::Instant;

use dfdx::prelude::*;
use dfdx_nn::*;

#[cfg(feature = "cuda")]
type Dev = Cuda;

#[cfg(not(feature = "cuda"))]
type Dev = Cpu;

type Model = BatchNorm2DConstConfig<512>;
type Dtype = f32;
type InputShape = Rank4<64, 512, 28, 28>;

fn main() {
    println!("Benchmarking `BatchNorm2D`");
    println!("Device {}", std::any::type_name::<Dev>());
    println!("Dtype {}", std::any::type_name::<Dtype>());
    println!("Input shape {}", std::any::type_name::<InputShape>());
    println!();

    let dev: Dev = Default::default();
    let mut m = dev.build_module_ext::<Dtype>(Model::default());
    let mut grads = m.alloc_grads();

    loop {
        let img: Tensor<InputShape, Dtype, _> = dev.sample_normal();

        let start = Instant::now();
        let _ = m.forward(img.clone());
        dev.synchronize();
        let infer_dur = start.elapsed();

        let start = Instant::now();
        let out = m.forward_mut(img.traced(grads));
        let loss = out.square().mean();
        dev.synchronize();
        let fwd_dur = start.elapsed();

        let start = Instant::now();
        grads = loss.backward();
        dev.synchronize();
        let bwd_dur = start.elapsed();
        println!("infer={infer_dur:?}, fwd={fwd_dur:?} bwd={bwd_dur:?}");
    }
}
