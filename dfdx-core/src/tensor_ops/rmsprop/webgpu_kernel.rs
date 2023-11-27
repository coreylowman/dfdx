use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::RMSpropKernel<E> for Webgpu {
    fn rmsprop_kernel(
        &self,
        cfg: &crate::prelude::RMSpropConfig,
        param: &mut Self::Vec,
        momentum: &mut Self::Vec,
        square_avg: &mut Self::Vec,
        grad_avg: &mut Self::Vec,
        grad: &Self::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
