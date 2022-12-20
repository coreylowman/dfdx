//! Intro to dfdx::arrays and dfdx::tensor

use dfdx::{
    shapes::{Rank1, Rank2, Rank3},
    tensor::{AsArray, Cpu, OnesTensor, SampleTensor, Tensor, TensorFromArray, ZerosTensor},
};

fn main() {
    // a device is required to create & modify tensors.
    // we will use the Cpu device here for simplicity
    let dev: Cpu = Default::default();

    // easily create tensors using the `TensorFromArray::tensor` method of devices
    // tensors are generic over the:
    // 1. Shape (in this case a rank 1 (1 dimension) array with 5 elements)
    // 2. Data type (in this case the default of `f32`)
    // 3. The device they are stored on (in this case the default of `Cpu`)
    // 4. A tape - see examples/04-gradients.rs
    let _: Tensor<Rank1<5>> = dev.tensor([1.0, 2.0, 3.0, 4.0, 5.0]);

    // You can also use [ZerosTensor::zeros] and [OnesTensor::ones] to create tensors
    // filled with the corresponding values.
    let _: Tensor<Rank2<2, 3>> = dev.zeros();
    let _: Tensor<Rank3<1, 2, 3>> = dev.ones();

    // each of the creation methods also supports specifying the shape on the function
    // note to change the dtype we specify the dtype as the 2nd generic parameter
    let _: Tensor<Rank2<2, 3>, f64> = dev.zeros();
    let _ = dev.ones::<Rank2<2, 3>>();

    // we can also create tensors filled with random values
    // from a normal distribution
    let _: Tensor<Rank3<2, 3, 4>> = dev.sample(rand_distr::StandardNormal);

    // or a uniform distribution
    let a: Tensor<Rank3<2, 3, 4>> = dev.sample(rand_distr::Uniform::new(0.0f32, 1.0));

    // use `AsArray::as_array` to get acces to the data as an array
    let a_data: [[[f32; 4]; 3]; 2] = a.array();
    println!("a={:?}", a_data);

    // you can clone() a tensor:
    let a_copy = a.clone();
    assert_eq!(a_copy.array(), a.array());
}
