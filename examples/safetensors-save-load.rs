//! Demonstrates how to save and load arrays with safetensors

#[cfg(feature = "safetensors")]
use dfdx::{
    shapes::{Dtype, HasShape, Rank0, Rank1, Rank2, Shape, Unit},
    tensor::{AsArray, AsVec, CopySlice, Cpu, Tensor, TensorFromArray, ZerosTensor},
};
#[cfg(feature = "safetensors")]
use safetensors::tensor::{
    serialize_to_file, Dtype as SDtype, SafeTensorError, SafeTensors, TensorView,
};

#[cfg(feature = "safetensors")]
fn view<'a, S, T>(tensor: &'a Tensor<S, f32>) -> TensorView<'a>
where
    T: Into<Vec<usize>>,
    S: Shape<Concrete = T>,
{
    let dtype = SDtype::F32;
    let shape = tensor.shape().concrete().into();
    let data = tensor.as_vec();
    let data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let data: &'static [u8] = Box::leak(Box::new(data));
    TensorView::new(dtype, shape, data)
}

#[cfg(feature = "safetensors")]
fn set_tensor<'a, S: Shape>(
    tensors: &SafeTensors<'a>,
    key: &str,
    out: &mut Tensor<S, f32>,
) -> Result<(), SafeTensorError> {
    let tensor = tensors.tensor(key)?;
    let v = tensor.data();
    let data: &[f32] = if (v.as_ptr() as usize) % 4 == 0 {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) }
    } else {
        let mut c = Vec::with_capacity(v.len() / 4);
        let mut i = 0;
        while i < v.len() {
            c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
            i += 4;
        }
        let c: &'static Vec<f32> = Box::leak(Box::new(c));
        c
    };
    out.copy_from(data);
    Ok(())
}

#[cfg(feature = "safetensors")]
fn main() {
    use std::collections::BTreeMap;

    let dev: Cpu = Default::default();

    let a = dev.tensor(1.234f32);
    let b = dev.tensor([1.0f32, 2.0, 3.0]);
    let c = dev.tensor([[1.0f32, 2.0, 3.0], [-1.0, -2.0, -3.0]]);

    let tensors = BTreeMap::from([
        ("a".to_string(), view(&a)),
        ("b".to_string(), view(&b)),
        ("c".to_string(), view(&c)),
    ]);

    let path = std::path::Path::new("out.safetensors");
    serialize_to_file(&tensors, &None, path);

    let mut a: Tensor<Rank0, f32, _> = dev.zeros();
    let mut b: Tensor<Rank1<3>, f32, _> = dev.zeros();
    let mut c: Tensor<Rank2<2, 3>, f32, _> = dev.zeros();

    let filename = "out.safetensors";
    let buffer = std::fs::read(filename).expect("Couldn't read file");
    let tensors = SafeTensors::deserialize(&buffer).expect("Couldn't read safetensors file");

    set_tensor(&tensors, "a", &mut a).expect("Loading a failed");
    set_tensor(&tensors, "b", &mut b).expect("Loading b failed");
    set_tensor(&tensors, "c", &mut c).expect("Loading c failed");

    assert_eq!(a.array(), 1.234);
    assert_eq!(b.array(), [1.0, 2.0, 3.0]);
    assert_eq!(c.array(), [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
}

#[cfg(not(feature = "safetensors"))]
fn main() {
    panic!("Use the 'safetensors' feature to run this example");
}
