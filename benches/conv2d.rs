#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "nightly")]
fn main() {
    use std::time::{Instant, Duration};

    use dfdx::prelude::*;

    #[cfg(feature = "cuda")]
    type Dev = Cuda;

    #[cfg(not(feature = "cuda"))]
    type Dev = Cpu;

    type Model = Conv2D<3, 16, 5>;
    type Dtype = f32;
    type InputShape = Rank4<4, 3, 640, 400>;

    println!("Benchmarking `Conv2D`");
    println!("Device {}", std::any::type_name::<Dev>());
    println!("Dtype {}", std::any::type_name::<Dtype>());
    println!("Input shape {}", std::any::type_name::<InputShape>());
    println!();

    let dev: Dev = Default::default();
    let mut m = dev.build_module::<Model, Dtype>();

    let mut sum = (0.0, 0.0);
    let mut sum_s = (0.0, 0.0);

    let tests = 1000;
    let tf = (tests-1) as f32;

    let true_s = Instant::now();
    for i in 0..tests {
        let img: Tensor<InputShape, Dtype, _, _> = dev.sample_normal().leaky_traced();

        let start = Instant::now();
        let out = m.forward_mut(img);
        let loss = out.square().mean();
        let fwd_dur = start.elapsed();

        let start = Instant::now();
        let _ = loss.backward();
        let bwd_dur = start.elapsed();
        println!("fwd={:?} bwd={:?}", fwd_dur, bwd_dur);

        if i != 0 {
            sum.0 += fwd_dur.as_secs_f32();
            sum_s.0 += fwd_dur.as_secs_f32().powi(2);

            sum.1 += bwd_dur.as_secs_f32();
            sum_s.1 += bwd_dur.as_secs_f32().powi(2);
        }
    }
    println!("{:?}", true_s.elapsed());
    // 152.2 without forward
    // 171.8 with forward

    // 65.9 without backward
    // 171.5 with backward
    // 86.6 with new backward

    // println!("{sum:?} {sum_s:?} {}", (sum_s.0/100. - sum.0*sum.0/10_000.).sqrt());
    println!("fwd_mean={:?} fwd_sd={:?}", Duration::from_secs_f32(sum.0/tf), Duration::from_secs_f32((sum_s.0/tf - sum.0*sum.0/tf/tf).sqrt()));
    println!("bwd_mean={:?} bwd_sd={:?}", Duration::from_secs_f32(sum.1/tf), Duration::from_secs_f32((sum_s.1/tf - sum.1*sum.1/tf/tf).sqrt()));
}

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("Run with `cargo +nightly run ...` to run this example.");
}
