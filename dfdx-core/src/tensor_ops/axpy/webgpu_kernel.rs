use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::AxpyKernel<E> for Webgpu {
    fn forward(
        &self,
        a: &mut Self::Vec,
        alpha: E,
        b: &Self::Vec,
        beta: E,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
