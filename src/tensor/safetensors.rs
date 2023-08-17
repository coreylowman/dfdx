use super::{CopySlice, Tensor};
use crate::shapes::{Dtype, Shape};
use safetensors::tensor::{Dtype as SDtype, SafeTensorError, SafeTensors};
use std::vec::Vec;

pub trait SafeDtype: Sized {
    type Array: IntoIterator<Item = u8>;

    fn from_le_bytes(bytes: &[u8], index: usize) -> Self;
    fn to_le_bytes(self) -> Self::Array;
    fn safe_dtype() -> SDtype;
}

#[cfg(feature = "f16")]
impl SafeDtype for crate::dtypes::f16 {
    type Array = [u8; 2];
    fn from_le_bytes(bytes: &[u8], index: usize) -> Self {
        Self::from_le_bytes(bytes[index..index + 2].try_into().unwrap())
    }

    fn to_le_bytes(self) -> Self::Array {
        self.to_le_bytes()
    }

    fn safe_dtype() -> SDtype {
        SDtype::F16
    }
}

impl SafeDtype for f32 {
    type Array = [u8; 4];
    fn from_le_bytes(bytes: &[u8], index: usize) -> Self {
        Self::from_le_bytes(bytes[index..index + 4].try_into().unwrap())
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
        Self::from_le_bytes(bytes[index..index + 8].try_into().unwrap())
    }

    fn safe_dtype() -> SDtype {
        SDtype::F64
    }

    fn to_le_bytes(self) -> Self::Array {
        self.to_le_bytes()
    }
}

impl<S: Shape, E: Dtype + SafeDtype, D: CopySlice<E>, T> Tensor<S, E, D, T> {
    /// Loads data from the [SafeTensors] Storage<E> with the given `key`
    pub fn load_safetensor(
        &mut self,
        tensors: &SafeTensors,
        key: &str,
    ) -> Result<(), SafeTensorError> {
        let tensor_view = tensors.tensor(key)?;
        let v = tensor_view.data();
        let num_bytes = std::mem::size_of::<E>();
        assert_eq!(
            tensor_view.shape(),
            self.shape.concrete().into(),
            "SafeTensors shape did not match tensor shape"
        );
        if (v.as_ptr() as usize) % num_bytes == 0 {
            // SAFETY This is safe because we just checked that this
            // was correctly aligned.
            let data: &[E] =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const E, v.len() / num_bytes) };
            self.copy_from(data);
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
