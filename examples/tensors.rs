#![allow(clippy::needless_range_loop)]
use dfdx::prelude::*;

fn main() {
    let a: Tensor2D<2, 3> = TensorCreator::zeros();

    // since add() expects tensors with the same size, we dont need a type for this
    let b = TensorCreator::ones();
    let c = add(a, &b);

    // tensors just store raw rust arrays, use `.data()` to access this.
    assert_eq!(c.data(), &[[1.0; 3]; 2]);

    // since we pass in an array, rust will figure out that we mean Tensor1D<5> since its an [f32; 5]
    let mut d = Tensor1D::new([1.0, 2.0, 3.0, 4.0, 5.0]);

    // use `.mut_data()` to access underlying mutable array. type is provided for readability
    let raw_data: &mut [f32; 5] = d.mut_data();
    for i in 0..5 {
        raw_data[i] *= 2.0;
    }
    assert_eq!(d.data(), &[2.0, 4.0, 6.0, 8.0, 10.0]);
}
