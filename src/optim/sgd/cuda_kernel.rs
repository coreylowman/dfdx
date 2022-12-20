use crate::{shapes::*, tensor::Cuda};

impl<E: Dtype> super::SgdKernel<E> for Cuda {
    fn update<S: Shape>(
        cfg: &super::SgdConfig<E>,
        param: &mut Self::Storage<S, E>,
        velocity: &mut Self::Storage<S, E>,
        grad: Self::Storage<S, E>,
    ) {
        todo!()
    }
}
