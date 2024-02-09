use super::{CopySlice, Tensor};
use crate::shapes::{Dtype, Shape};
use safetensors::tensor::{SafeTensorError, SafeTensors};
use std::vec::Vec;

impl<S: Shape, E: Dtype, D: CopySlice<E>, T> Tensor<S, E, D, T> {
    /// Loads data from the [SafeTensors] `Storage<E>` with the given `key`
    pub fn load_safetensor<F: FnMut(String) -> String>(
        &mut self,
        tensors: &SafeTensors,
        key: &str,
        skip_missing: bool,
        key_map: &mut F,
    ) -> Result<(), SafeTensorError> {
        let key = key_map(key.to_string());
        let tensor_view = match tensors.tensor(&key) {
            Ok(ok) => ok,
            Err(safetensors::SafeTensorError::TensorNotFound(_name)) if skip_missing => {
                return Ok(());
            }
            Err(e) => return Err(e),
        };
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
                c.push(E::from_le_bytes(&v[i..i + num_bytes]));
                i += num_bytes;
            }
            self.copy_from(&c);
        };
        Ok(())
    }
}
