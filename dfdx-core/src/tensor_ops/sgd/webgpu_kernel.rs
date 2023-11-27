use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::SgdKernel<E> for Webgpu {
    fn sgd_kernel(
        &self,
        cfg: &crate::prelude::SgdConfig,
        param: &mut Self::Vec,
        velocity: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
