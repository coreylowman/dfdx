//! Demonstrates how to save and load arrays with tensors

#[cfg(feature = "numpy")]
fn main() {
    use dfdx::{
        shapes::{Rank0, Rank1, Rank2},
        tensor::{AsArray, AutoDevice, Tensor, TensorFrom, ZerosTensor},
    };

    let dev = AutoDevice::default();

    dev.tensor(1.234f32)
        .save_to_npy("0d-rs.npy")
        .expect("Saving failed");

    dev.tensor([1.0f32, 2.0, 3.0])
        .save_to_npy("1d-rs.npy")
        .expect("Saving failed");

    dev.tensor([[1.0f32, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        .save_to_npy("2d-rs.npy")
        .expect("Saving failed");

    let mut a: Tensor<Rank0, f32, _> = dev.zeros();
    a.load_from_npy("0d-rs.npy").expect("Loading failed");
    assert_eq!(a.array(), 1.234);

    let mut b: Tensor<Rank1<3>, f32, _> = dev.zeros();
    b.load_from_npy("1d-rs.npy").expect("Loading failed");
    assert_eq!(b.array(), [1.0, 2.0, 3.0]);

    let mut c: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();
    c.load_from_npy("2d-rs.npy").expect("Loading failed");
    assert_eq!(c.array(), [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);

    println!("Tensors were stored and loaded successfully");
}

#[cfg(not(feature = "numpy"))]
fn main() {
    panic!("Use the 'numpy' feature to run this example");
}
