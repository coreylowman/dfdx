use crate::{shapes::Shape, tensor::Cuda};

impl super::AdamKernel<f32> for Cuda {
    fn update<S: Shape>(
        t: i32,
        cfg: &super::AdamConfig<f32>,
        param: &mut Self::Storage<S, f32>,
        moment1: &mut Self::Storage<S, f32>,
        moment2: &mut Self::Storage<S, f32>,
        grad: Self::Storage<S, f32>,
    ) {
        todo!()
    }
}
