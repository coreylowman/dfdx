use crate::prelude::*;

/// **Requires Nightly** Flattens 3d tensors to 1d, and 4d tensors to 2d.
///
/// Specifically:
/// ```ignore
/// # use dfdx::prelude::*;
/// let _: Tensor1D<{3 * 5 * 7}> = FlattenImage.forward(Tensor3D::<3, 5, 7>::zeros());
/// let _: Tensor2D<8, {3 * 5 * 7}> = FlattenImage.forward(Tensor4D::<8, 3, 5, 7>::zeros());
/// ```
#[derive(Default, Clone, Copy)]
pub struct FlattenImage;

impl ResetParams for FlattenImage {
    /// Does nothing.
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {}
}

impl CanUpdateWithGradients for FlattenImage {
    /// Does nothing.
    fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
}

impl SaveToNpz for FlattenImage {}
impl LoadFromNpz for FlattenImage {}

impl<const M: usize, const N: usize, const O: usize, H: Tape> Module<Tensor3D<M, N, O, H>>
    for FlattenImage
where
    Assert<{ M * N * O == (M * N * O) }>: ConstTrue,
{
    type Output = Tensor1D<{ M * N * O }, H>;
    fn forward(&self, input: Tensor3D<M, N, O, H>) -> Self::Output {
        Reshape::<Self::Output>::reshape(input)
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, H: Tape>
    Module<Tensor4D<M, N, O, P, H>> for FlattenImage
where
    Assert<{ M * N * O * P == M * (N * O * P) }>: ConstTrue,
{
    type Output = Tensor2D<M, { N * O * P }, H>;
    fn forward(&self, input: Tensor4D<M, N, O, P, H>) -> Self::Output {
        Reshape::<Self::Output>::reshape(input)
    }
}

impl<T> ModuleMut<T> for FlattenImage
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;

    fn forward_mut(&mut self, t: T) -> Self::Output {
        self.forward(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flattens() {
        let _: Tensor1D<{ 15 * 10 * 5 }> = FlattenImage.forward(Tensor3D::<15, 10, 5>::zeros());
        let _: Tensor2D<5, 24> = FlattenImage.forward(Tensor4D::<5, 4, 3, 2>::zeros());
    }
}
