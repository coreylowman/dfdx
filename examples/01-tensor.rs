//! Intro to dfdx::tensor

use rand::thread_rng;

use dfdx::arrays::HasArrayData;
use dfdx::tensor::{tensor, Tensor1D, Tensor2D, Tensor3D, TensorCreator};

fn main() {
    // easily create tensors using the `tensor` function
    let _: Tensor1D<5> = tensor([1.0, 2.0, 3.0, 4.0, 5.0]);

    // you can also use [TensorCreator::new]
    let _: Tensor1D<5> = TensorCreator::new([1.0, 2.0, 3.0, 4.0, 5.0]);

    // [TensorCreator] has other helpful methods such as all zeros and all ones
    let _: Tensor2D<2, 3> = TensorCreator::zeros();
    let _: Tensor2D<2, 3> = TensorCreator::ones();

    // we can also create random tensors
    let mut rng = thread_rng();
    let a: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);

    // use `.data()` to access the underlying array
    let a_data: &[[[f32; 4]; 3]; 2] = a.data();
    println!("a={:?}", a_data);

    // you can clone() a tensor:
    let a_copy = a.clone();
    assert_eq!(a_copy.data(), a.data());
}
