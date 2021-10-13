use stag::nn::Linear;
use stag::prelude::*;

fn main() {
    let mut model = Linear::<10, 5>::default();

    // create a 1x10 tensor filled with 0s
    let mut a: Tensor1D<10> = Tensor1D::default();

    // create a 64x10 tensor filled with 0s
    let mut b: Tensor2D<64, 10> = Tensor2D::default();

    // yay both of these work!
    let y = model.forward(&mut a);
    let z = model.forward(&mut b);
}
