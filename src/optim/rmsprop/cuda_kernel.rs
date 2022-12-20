use crate::tensor::Cuda;

impl super::RMSpropKernel<f32> for Cuda {
    fn update<S: crate::shapes::Shape>(
        cfg: &super::RMSpropConfig<f32>,
        param: &mut Self::Storage<S, f32>,
        momentum: &mut Self::Storage<S, f32>,
        square_avg: &mut Self::Storage<S, f32>,
        grad_avg: &mut Self::Storage<S, f32>,
        grad: Self::Storage<S, f32>,
    ) {
        todo!()
    }
}
