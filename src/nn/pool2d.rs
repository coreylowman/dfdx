use super::{Module, ModuleMut, ResetParams};
use crate::gradients::*;
#[cfg(feature = "nightly")]
use crate::tensor::*;
use rand::Rng;

/// Average pool with 2d kernel that operates on images (3d) and batches of images (4d).
/// Each patch reduces to the average of the values in the patch.
///
/// Generics:
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
#[derive(Debug, Default, Clone)]
pub struct AvgPool2D<const KERNEL_SIZE: usize, const STRIDE: usize = 1, const PADDING: usize = 0>;

/// Max pool with 2d kernel that operates on images (3d) and batches of images (4d).
/// Each patch reduces to the maximum value in that patch.
///
/// Generics:
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
#[derive(Debug, Default, Clone)]
pub struct MaxPool2D<const KERNEL_SIZE: usize, const STRIDE: usize = 1, const PADDING: usize = 0>;

/// Minimum pool with 2d kernel that operates on images (3d) and batches of images (4d).
/// Each patch reduces to the minimum of the values in the patch.
///
/// Generics:
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
#[derive(Debug, Default, Clone)]
pub struct MinPool2D<const KERNEL_SIZE: usize, const STRIDE: usize = 1, const PADDING: usize = 0>;

macro_rules! impl_pools {
    ($PoolTy:tt, $Method:ident) => {
        impl<const K: usize, const S: usize, const P: usize> CanUpdateWithGradients
            for $PoolTy<K, S, P>
        {
            fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
        }

        impl<const K: usize, const S: usize, const P: usize> ResetParams for $PoolTy<K, S, P> {
            fn reset_params<R: Rng>(&mut self, _: &mut R) {}
        }

        #[cfg(feature = "nightly")]
        impl<
                const K: usize,
                const S: usize,
                const P: usize,
                const C: usize,
                const H: usize,
                const W: usize,
                T: Tape,
            > Module<Tensor3D<C, H, W, T>> for $PoolTy<K, S, P>
        where
            [[(); (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]:,
        {
            type Output = Tensor3D<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

            fn forward(&self, x: Tensor3D<C, H, W, T>) -> Self::Output {
                x.$Method::<K, S, P>()
            }
        }

        #[cfg(feature = "nightly")]
        impl<
                const K: usize,
                const S: usize,
                const P: usize,
                const B: usize,
                const C: usize,
                const H: usize,
                const W: usize,
                T: Tape,
            > Module<Tensor4D<B, C, H, W, T>> for $PoolTy<K, S, P>
        where
            [[(); (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]:,
        {
            type Output =
                Tensor4D<B, C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }, T>;

            fn forward(&self, x: Tensor4D<B, C, H, W, T>) -> Self::Output {
                x.$Method::<K, S, P>()
            }
        }

        impl<T, const K: usize, const S: usize, const P: usize> ModuleMut<T> for $PoolTy<K, S, P>
        where
            Self: Module<T>,
        {
            type Output = <Self as Module<T>>::Output;
            fn forward_mut(&mut self, input: T) -> Self::Output {
                self.forward(input)
            }
        }
    };
}

impl_pools!(AvgPool2D, avg2d);
impl_pools!(MaxPool2D, max2d);
impl_pools!(MinPool2D, min2d);

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_forward_3d_sizes() {
        type Img = Tensor3D<3, 10, 10>;
        let _: Tensor3D<3, 8, 8> = MaxPool2D::<3>::default().forward(Img::zeros());
        let _: Tensor3D<3, 9, 9> = MaxPool2D::<2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 7, 7> = MaxPool2D::<4>::default().forward(Img::zeros());
        let _: Tensor3D<3, 4, 4> = MaxPool2D::<3, 2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 3, 3> = MaxPool2D::<3, 3>::default().forward(Img::zeros());
        let _: Tensor3D<3, 10, 10> = MaxPool2D::<3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor3D<3, 12, 12> = MaxPool2D::<3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 6, 6> = MaxPool2D::<3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_max_forward_4d_sizes() {
        type Img = Tensor4D<5, 3, 10, 10>;
        let _: Tensor4D<5, 3, 7, 7> = MaxPool2D::<4>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 8, 8> = MaxPool2D::<3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 9, 9> = MaxPool2D::<2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 4, 4> = MaxPool2D::<3, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 3, 3> = MaxPool2D::<3, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 10, 10> = MaxPool2D::<3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 12, 12> = MaxPool2D::<3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 6, 6> = MaxPool2D::<3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_max_tuple_pool_sizes() {
        type A = MaxPool2D<3>;
        type B = MaxPool2D<1, 1, 1>;
        type Img = Tensor3D<1, 10, 10>;

        let _: Tensor3D<1, 6, 6> = <(A, A)>::default().forward(Tensor3D::<1, 10, 10>::zeros());
        let _: Tensor3D<1, 8, 8> = <(A, A, B)>::default().forward(Img::zeros());
    }

    #[test]
    fn test_min_forward_3d_sizes() {
        type Img = Tensor3D<3, 10, 10>;
        let _: Tensor3D<3, 8, 8> = MinPool2D::<3>::default().forward(Img::zeros());
        let _: Tensor3D<3, 9, 9> = MinPool2D::<2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 7, 7> = MinPool2D::<4>::default().forward(Img::zeros());
        let _: Tensor3D<3, 4, 4> = MinPool2D::<3, 2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 3, 3> = MinPool2D::<3, 3>::default().forward(Img::zeros());
        let _: Tensor3D<3, 10, 10> = MinPool2D::<3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor3D<3, 12, 12> = MinPool2D::<3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 6, 6> = MinPool2D::<3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_min_forward_4d_sizes() {
        type Img = Tensor4D<5, 3, 10, 10>;
        let _: Tensor4D<5, 3, 7, 7> = MinPool2D::<4>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 8, 8> = MinPool2D::<3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 9, 9> = MinPool2D::<2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 4, 4> = MinPool2D::<3, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 3, 3> = MinPool2D::<3, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 10, 10> = MinPool2D::<3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 12, 12> = MinPool2D::<3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 6, 6> = MinPool2D::<3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_min_tuple_pool_sizes() {
        type A = MinPool2D<3>;
        type B = MinPool2D<1, 1, 1>;
        type Img = Tensor3D<1, 10, 10>;

        let _: Tensor3D<1, 6, 6> = <(A, A)>::default().forward(Tensor3D::<1, 10, 10>::zeros());
        let _: Tensor3D<1, 8, 8> = <(A, A, B)>::default().forward(Img::zeros());
    }

    #[test]
    fn test_avg_forward_3d_sizes() {
        type Img = Tensor3D<3, 10, 10>;
        let _: Tensor3D<3, 8, 8> = AvgPool2D::<3>::default().forward(Img::zeros());
        let _: Tensor3D<3, 9, 9> = AvgPool2D::<2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 7, 7> = AvgPool2D::<4>::default().forward(Img::zeros());
        let _: Tensor3D<3, 4, 4> = AvgPool2D::<3, 2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 3, 3> = AvgPool2D::<3, 3>::default().forward(Img::zeros());
        let _: Tensor3D<3, 10, 10> = AvgPool2D::<3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor3D<3, 12, 12> = AvgPool2D::<3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor3D<3, 6, 6> = AvgPool2D::<3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_avg_forward_4d_sizes() {
        type Img = Tensor4D<5, 3, 10, 10>;
        let _: Tensor4D<5, 3, 7, 7> = AvgPool2D::<4>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 8, 8> = AvgPool2D::<3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 9, 9> = AvgPool2D::<2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 4, 4> = AvgPool2D::<3, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 3, 3> = AvgPool2D::<3, 3>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 10, 10> = AvgPool2D::<3, 1, 1>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 12, 12> = AvgPool2D::<3, 1, 2>::default().forward(Img::zeros());
        let _: Tensor4D<5, 3, 6, 6> = AvgPool2D::<3, 2, 2>::default().forward(Img::zeros());
    }

    #[test]
    fn test_avg_tuple_pool_sizes() {
        type A = AvgPool2D<3>;
        type B = AvgPool2D<1, 1, 1>;
        type Img = Tensor3D<1, 10, 10>;

        let _: Tensor3D<1, 6, 6> = <(A, A)>::default().forward(Tensor3D::<1, 10, 10>::zeros());
        let _: Tensor3D<1, 8, 8> = <(A, A, B)>::default().forward(Img::zeros());
    }
}
