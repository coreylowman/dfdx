use super::{LoadFromNpz, SaveToNpz};
use super::{Module, ResetParams};
use crate::gradients::*;
use crate::tensor::*;
use rand::Rng;

pub struct AvgPool2D<const KERNEL_SIZE: usize, const STRIDE: usize = 1, const PADDING: usize = 0>;

impl<const K: usize, const S: usize, const P: usize> CanUpdateWithGradients for AvgPool2D<K, S, P> {
    fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
}

impl<const K: usize, const S: usize, const P: usize> ResetParams for AvgPool2D<K, S, P> {
    fn reset_params<R: Rng>(&mut self, _: &mut R) {}
}

impl<const K: usize, const S: usize, const P: usize> SaveToNpz for AvgPool2D<K, S, P> {}
impl<const K: usize, const S: usize, const P: usize> LoadFromNpz for AvgPool2D<K, S, P> {}

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        T: Tape,
    > Module<Tensor3D<C, H, W, T>> for AvgPool2D<K, S, P>
where
    [(); (W + 2 * P - K) / S + 1]:,
    [(); (H + 2 * P - K) / S + 1]:,
{
    type Output = Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

    fn forward(&self, x: Tensor3D<C, H, W, T>) -> Self::Output {
        x.avg2d::<K, S, P>()
    }
}

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        const B: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        T: Tape,
    > Module<Tensor4D<B, C, H, W, T>> for AvgPool2D<K, S, P>
where
    [(); (W + 2 * P - K) / S + 1]:,
    [(); (H + 2 * P - K) / S + 1]:,
{
    type Output = Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

    fn forward(&self, x: Tensor4D<B, C, H, W, T>) -> Self::Output {
        x.avg2d::<K, S, P>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_3d_sizes() {
        todo!();
    }

    #[test]
    fn test_forward_4d_sizes() {
        todo!();
    }
}
