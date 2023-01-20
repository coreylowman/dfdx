//! Demonstrates how to save and load arrays with safetensors

#[cfg(feature = "safetensors")]
fn main() {
    use dfdx::{
        shapes::{Rank0, Rank1, Rank2},
        tensor::safetensors::Writer,
        tensor::{AsArray, Cpu, Tensor, TensorFromArray, ZerosTensor},
    };
    use safetensors::tensor::SafeTensors;
    let dev: Cpu = Default::default();

    let a = dev.tensor(1.234f32);
    let b = dev.tensor([1.0f32, 2.0, 3.0]);
    let c = dev.tensor([[1.0f32, 2.0, 3.0], [-1.0, -2.0, -3.0]]);

    let path = std::path::Path::new("out.safetensors");

    Writer::new()
        .add("a".to_string(), a)
        .add("b".to_string(), b)
        .add("c".to_string(), c)
        .save(path)
        .unwrap();

    let mut a: Tensor<Rank0, f32, _> = dev.zeros();
    let mut b: Tensor<Rank1<3>, f32, _> = dev.zeros();
    let mut c: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();

    let filename = "out.safetensors";
    let buffer = std::fs::read(filename).expect("Couldn't read file");
    let tensors = SafeTensors::deserialize(&buffer).expect("Couldn't read safetensors file");
    a.load(&tensors, "a").expect("Loading a failed");
    b.load(&tensors, "b").expect("Loading b failed");
    c.load(&tensors, "c").expect("Loading c failed");

    assert_eq!(a.array(), 1.234);
    assert_eq!(b.array(), [1.0, 2.0, 3.0]);
    assert_eq!(c.array(), [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
}

#[cfg(not(feature = "safetensors"))]
fn main() {
    panic!("Use the 'safetensors' feature to run this example");
}
