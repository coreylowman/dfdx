use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::AdamKernel<E> for Webgpu {
    fn adam_kernel(
        &self,
        t: i32,
        cfg: &crate::prelude::AdamConfig,
        param: &mut Self::Vec,
        moment1: &mut Self::Vec,
        moment2: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
