use dfdx::prelude::{Device, Dim, Dtype, MeanTo, Tape, Tensor};

#[derive(Default, Debug, Clone, Copy, crate::CustomModule)]
pub struct AvgPoolGlobal;

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(C, H, W), E, D, T>> for AvgPoolGlobal
{
    type Output = Tensor<(C,), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_mean()
    }
}

impl<B: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(B, C, H, W), E, D, T>> for AvgPoolGlobal
{
    type Output = Tensor<(B, C), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, D::Err> {
        input.try_mean()
    }
}
