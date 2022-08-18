use crate::prelude::*;

/// **Requires Nightly** Flattens anything above 2 dimensions to 2d. For example a 3d tensor
/// of shape (M, N, O) will be flattened to a 2d tensor of shape (M, N * O)
#[derive(Default, Clone, Copy)]
pub struct Flatten;

impl ResetParams for Flatten {
    /// Does nothing.
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {}
}

impl CanUpdateWithGradients for Flatten {
    /// Does nothing.
    fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
}

impl SaveToNpz for Flatten {}
impl LoadFromNpz for Flatten {}

impl<H: Tape> Module<Tensor0D<H>> for Flatten {
    type Output = Tensor0D<H>;
    fn forward(&self, input: Tensor0D<H>) -> Self::Output {
        input
    }
}

impl<const M: usize, H: Tape> Module<Tensor1D<M, H>> for Flatten {
    type Output = Tensor1D<M, H>;
    fn forward(&self, input: Tensor1D<M, H>) -> Self::Output {
        input
    }
}

impl<const M: usize, const N: usize, H: Tape> Module<Tensor2D<M, N, H>> for Flatten {
    type Output = Tensor2D<M, N, H>;
    fn forward(&self, input: Tensor2D<M, N, H>) -> Self::Output {
        input
    }
}

impl<const M: usize, const N: usize, const O: usize, H: Tape> Module<Tensor3D<M, N, O, H>>
    for Flatten
where
    Assert<{ M * N * O == M * (N * O) }>: ConstTrue,
{
    type Output = Tensor2D<M, { N * O }, H>;
    fn forward(&self, input: Tensor3D<M, N, O, H>) -> Self::Output {
        Reshape::<Self::Output>::reshape(input)
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, H: Tape>
    Module<Tensor4D<M, N, O, P, H>> for Flatten
where
    Assert<{ M * N * O * P == M * (N * O * P) }>: ConstTrue,
{
    type Output = Tensor2D<M, { N * O * P }, H>;
    fn forward(&self, input: Tensor4D<M, N, O, P, H>) -> Self::Output {
        Reshape::<Self::Output>::reshape(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flattens() {
        let _: Tensor0D = Flatten.forward(Tensor0D::zeros());
        let _: Tensor1D<5> = Flatten.forward(Tensor1D::<5>::zeros());
        let _: Tensor2D<10, 5> = Flatten.forward(Tensor2D::<10, 5>::zeros());
        let _: Tensor2D<15, 50> = Flatten.forward(Tensor3D::<15, 10, 5>::zeros());
        let _: Tensor2D<5, 24> = Flatten.forward(Tensor4D::<5, 4, 3, 2>::zeros());
    }
}
