//! Intro to dfdx::devices and dfdx::tensor

use dfdx::devices::{AsArray, Cpu, Ones, Randn, Zeros};
use dfdx::tensor::{Tensor1D, Tensor2D, Tensor3D, TensorFromArray};

fn main() {
    // a device is required to create & modify tensors.
    // we will use the Cpu device here for simplicity
    let dev: Cpu = Default::default();

    // easily create tensors using the `TensorSugar::tensor` method
    // notice that Tensor's are generic over the device they are on.
    let _: Tensor1D<5, Cpu> = dev.tensor([1.0, 2.0, 3.0, 4.0, 5.0]);

    // You can also use [Zeros::zeros] and [Ones::ones] to create tensors
    // filled with the corresponding values.
    let _: Tensor2D<2, 3, Cpu> = dev.zeros();
    let _: Tensor2D<2, 3, _> = dev.ones();

    // we can also create tensors filled with random values
    // from a normal distribution
    let a: Tensor3D<2, 3, 4, _> = dev.randn();

    // use `AsArray::as_array` to get acces to the data as an array
    let a_data: [[[f32; 4]; 3]; 2] = a.as_array();
    println!("a={:?}", a_data);

    // you can clone() a tensor:
    let a_copy = a.clone();
    assert_eq!(a_copy.as_array(), a.as_array());
}
