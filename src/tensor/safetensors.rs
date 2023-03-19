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

#[derive(Debug)]
pub enum Error {
    SafeTensorError(SafeTensorError),
    MismatchedDimension((Vec<usize>, Vec<usize>)),
    IoError(std::io::Error),
}

impl From<SafeTensorError> for Error {
    fn from(safe_error: SafeTensorError) -> Error {
        Error::SafeTensorError(safe_error)
    }
}
impl From<std::io::Error> for Error {
    fn from(io_error: std::io::Error) -> Error {
        Error::IoError(io_error)
    }
}

impl<S: Shape, E: Dtype + SafeDtype, D: CopySlice<E>, T> Tensor<S, E, D, T> {
    /// Loads data from the [SafeTensors] storage with the given `key`
    pub fn load_safetensor(&mut self, tensors: &SafeTensors, key: &str) -> Result<(), Error> {
        let tensor = tensors.tensor(key)?;
        let v = tensor.data();
        let num_bytes = std::mem::size_of::<E>();
        if tensor.shape() != self.shape.concrete().into() {
            return Err(Error::MismatchedDimension((
                tensor.shape().to_vec(),
                self.shape.concrete().into(),
            )));
        }
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
