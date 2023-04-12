//! Intro to dfdx::arrays and dfdx::tensor

use dfdx::{
    shapes::{Const, Rank1, Rank2, Rank3},
    tensor::{AsArray, OnesTensor, SampleTensor, Tensor, TensorFrom, ZerosTensor},
    tensor_ops::RealizeTo,
};

#[cfg(not(feature = "cuda"))]
type Device = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
type Device = dfdx::tensor::Cuda;

fn main() {
    // a device is required to create & modify tensors.
    let dev: Device = Device::default();

    // easily create tensors using the `TensorFromArray::tensor` method of devices
    // tensors are generic over the:
    // 1. Shape (in this case a rank 1 (1 dimension) array with 5 elements)
    // 2. Data type (in this case the default of `f32`)
    // 3. The device they are stored on (in this case the default of `Cpu`)
    // 4. A tape - see examples/04-gradients.rs
    let _: Tensor<Rank1<5>, f32, Device> = dev.tensor([1.0, 2.0, 3.0, 4.0, 5.0]);

    // You can also use [ZerosTensor::zeros] and [OnesTensor::ones] to create tensors
    // filled with the corresponding values.
    let _: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
    let _: Tensor<Rank3<1, 2, 3>, f32, _> = dev.ones();

    // Use zeros_like & ones_like variants to create runtime sized tensors
    let _: Tensor<Rank2<2, 3>, f32, _> = dev.zeros_like(&(Const::<2>, Const::<3>));
    let _: Tensor<(usize, Const<3>), f32, _> = dev.zeros_like(&(2, Const::<3>));
    let _: Tensor<(usize, usize), f32, _> = dev.zeros_like(&(2, 4));
    let _: Tensor<(usize, usize, usize), f32, _> = dev.ones_like(&(3, 4, 5));
    let _: Tensor<(usize, usize, Const<5>), f32, _> = dev.ones_like(&(3, 4, Const));

    // `realize` method helps us move between dynamic and known size for the dimensions,
    // if the conversion is incompatible, it may result in runtime error
    let a: Tensor<(usize, usize), f32, _> = dev.zeros_like(&(2, 3));
    let _: Tensor<(usize, Const<3>), f32, _> = a.realize().expect("`a` should have 3 columns");

    // each of the creation methods also supports specifying the shape on the function
    // note to change the dtype we specify the dtype as the 2nd generic parameter
    let _: Tensor<Rank2<2, 3>, f64, _> = dev.zeros();
    let _: Tensor<Rank2<2, 3>, usize, _> = dev.zeros();
    let _: Tensor<Rank2<2, 3>, i16, _> = dev.zeros();

    // we can also create tensors filled with random values
    // from a normal distribution
    let _: Tensor<Rank3<2, 3, 4>, f32, Device> = dev.sample_normal();

    // or a uniform distribution
    let _: Tensor<Rank3<2, 3, 4>, f32, Device> = dev.sample_uniform();

    // or whatever distribution you want to use!
    let a: Tensor<Rank3<2, 3, 4>, f32, Device> = dev.sample(rand_distr::Uniform::new(-1.0, 1.0));

    // the random methods also have _like variants
    let _: Tensor<(usize, usize), f32, _> = dev.sample_uniform_like(&(1, 2));
    let _: Tensor<(usize, usize, usize), f32, _> = dev.sample_normal_like(&(1, 2, 3));
    let _: Tensor<(usize, usize, usize, usize), u64, _> =
        dev.sample_like(&(1, 2, 3, 4), rand_distr::StandardGeometric);

    // a more advanced use case involves creating a tensor from a vec
    let _: Tensor<Rank3<1, 2, 3>, f32, _> = dev.tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let _: Tensor<(usize,), f32, _> = dev.tensor((vec![1.0, 2.0], (2,)));

    // use `AsArray::as_array` to get access to the data as an array
    let a_data: [[[f32; 4]; 3]; 2] = a.array();
    println!("a={a_data:?}");

    // or as_vec to get a contiguous vec
    let a_data: Vec<f32> = a.as_vec();
    println!("a={a_data:?}");

    // you can clone() a tensor:
    let a_copy = a.clone();
    assert_eq!(a_copy.array(), a.array());
}
