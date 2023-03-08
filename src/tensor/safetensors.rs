use super::{CopySlice, DeviceStorage, Tensor};
use crate::shapes::{Dtype, Shape};
use safetensors::tensor::{Dtype as SDtype, SafeTensorError, SafeTensors};
use std::vec::Vec;

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
    pub fn load_safetensor<'a>(
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
