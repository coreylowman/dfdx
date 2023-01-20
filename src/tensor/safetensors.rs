use super::{CopySlice, DeviceStorage, Tensor};
use crate::{
    shapes::{Dtype, HasShape, Shape},
    tensor::AsVec,
};
use no_std_compat::{collections::BTreeMap, path::Path, string::String, vec::Vec};
use safetensors::tensor::{
    serialize_to_file, Dtype as SDtype, SafeTensorError, SafeTensors, TensorView,
};

struct TensorData {
    dtype: SDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

pub struct Writer {
    tensors: BTreeMap<String, TensorData>,
}

impl Writer {
    pub fn new() -> Self {
        let tensors = BTreeMap::new();
        Self { tensors }
    }

    pub fn add<S, T, E: Dtype + SafeDtype>(mut self, key: String, tensor: Tensor<S, E>) -> Self
    where
        T: Into<Vec<usize>>,
        S: Shape<Concrete = T>,
    {
        let dtype = E::safe_dtype();
        let shape = tensor.shape().concrete().into();
        let data = tensor.as_vec();
        let data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tdata = TensorData { dtype, shape, data };
        self.tensors.insert(key, tdata);
        self
    }

    pub fn save(&self, path: &Path) -> Result<(), SafeTensorError> {
        let views: BTreeMap<String, TensorView> = self
            .tensors
            .iter()
            .map(|(k, tensor)| {
                (
                    k.clone(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data),
                )
            })
            .collect();
        serialize_to_file(&views, &None, path)
    }
}

pub trait SafeDtype: Sized {
    type Array: IntoIterator<Item = u8>;

    fn from_le_bytes(bytes: &[u8], index: usize) -> Self;
    fn to_le_bytes(self) -> Self::Array;
    fn safe_dtype() -> SDtype;
}

impl SafeDtype for f32 {
    type Array = [u8; 4];
    fn from_le_bytes(bytes: &[u8], index: usize) -> Self {
        Self::from_le_bytes([
            bytes[index],
            bytes[index + 1],
            bytes[index + 2],
            bytes[index + 3],
        ])
    }

    fn to_le_bytes(self) -> Self::Array {
        self.to_le_bytes()
    }

    fn safe_dtype() -> SDtype {
        SDtype::F32
    }
}

impl SafeDtype for f64 {
    type Array = [u8; 8];
    fn from_le_bytes(bytes: &[u8], index: usize) -> Self {
        Self::from_le_bytes([
            bytes[index],
            bytes[index + 1],
            bytes[index + 2],
            bytes[index + 3],
            bytes[index + 4],
            bytes[index + 5],
            bytes[index + 6],
            bytes[index + 7],
        ])
    }

    fn safe_dtype() -> SDtype {
        SDtype::F64
    }

    fn to_le_bytes(self) -> Self::Array {
        self.to_le_bytes()
    }
}

impl<S: Shape, E: Dtype + SafeDtype, D: DeviceStorage + CopySlice<E>, T> Tensor<S, E, D, T> {
    pub fn safetensors_writer() -> Writer {
        Writer::new()
    }

    pub fn load<'a>(
        &mut self,
        tensors: &SafeTensors<'a>,
        key: &str,
    ) -> Result<(), SafeTensorError> {
        let tensor = tensors.tensor(key)?;
        let v = tensor.data();
        let num_bytes = std::mem::size_of::<E>();
        if (v.as_ptr() as usize) % num_bytes == 0 {
            // SAFETY This is safe because we just checked that this
            // was correctly aligned.
            let data: &[E] =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const E, v.len() / num_bytes) };
            self.copy_from(&data);
        } else {
            let mut c = Vec::with_capacity(v.len() / num_bytes);
            let mut i = 0;
            while i < v.len() {
                c.push(E::from_le_bytes(v, i));
                i += num_bytes;
            }
            self.copy_from(&c);
        };
        Ok(())
    }
}
