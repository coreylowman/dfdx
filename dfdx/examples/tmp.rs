use dfdx::prelude::*;

fn main() {
    let dev = Cpu::default();
    let x: Tensor<(usize,), f32, Cpu> = dev.ones_like(&(0,));
    // This line crashes.
    println!("{:?}", x.as_vec());
}
