use super::{LoadFromNpz, SaveToNpz};
use super::{Module, ResetParams};
use crate::gradients::*;
use crate::tensor::*;
use crate::tensor_ops::{min2d, min2d_batched};
use rand::Rng;

pub struct MinPool2D<const KERNEL_SIZE: usize, const STRIDE: usize = 1, const PADDING: usize = 0>;

impl<const K: usize, const S: usize, const P: usize> CanUpdateWithGradients for MinPool2D<K, S, P> {
    fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
}

impl<const K: usize, const S: usize, const P: usize> ResetParams for MinPool2D<K, S, P> {
    fn reset_params<R: Rng>(&mut self, _: &mut R) {}
}

impl<const K: usize, const S: usize, const P: usize> SaveToNpz for MinPool2D<K, S, P> {}
impl<const K: usize, const S: usize, const P: usize> LoadFromNpz for MinPool2D<K, S, P> {}

impl<
        T: Tape,
        const C: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const H: usize,
        const W: usize,
    > Module<Tensor3D<C, H, W, T>> for MinPool2D<K, S, P>
where
    [(); (W + 2 * P - K) / S + 1]:,
    [(); (H + 2 * P - K) / S + 1]:,
{
    type Output = Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

    fn forward(&self, x: Tensor3D<C, H, W, T>) -> Self::Output {
        min2d::<T, C, K, S, P, H, W>(x)
    }
}

impl<
        T: Tape,
        const B: usize,
        const C: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const H: usize,
        const W: usize,
    > Module<Tensor4D<B, C, H, W, T>> for MinPool2D<K, S, P>
where
    [(); (W + 2 * P - K) / S + 1]:,
    [(); (H + 2 * P - K) / S + 1]:,
{
    type Output = Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

    fn forward(&self, x: Tensor4D<B, C, H, W, T>) -> Self::Output {
        min2d_batched::<T, B, C, K, S, P, H, W>(x)
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
