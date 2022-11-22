use crate::{
    arrays::Shape,
    devices::{
        device::{HasErr, UnaryKernel},
        unary_ops, Device,
    },
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::try_unary_op;

mod internals {
    use super::*;

    pub trait PoolTo<T, Kind, const K: usize, const S: usize, const P: usize>: HasErr {
        fn try_pool_to(self) -> Result<T, Self::Err>;
    }

    impl<
            Src: Shape,
            Dst: Shape,
            Kind: 'static + Default + Clone,
            D: Device,
            T: Tape<D>,
            const K: usize,
            const S: usize,
            const P: usize,
        > PoolTo<Tensor<Dst, f32, D, T>, Kind, K, S, P> for Tensor<Src, f32, D, T>
    where
        D: UnaryKernel<unary_ops::Pool2D<Kind, K, S, P>, Src, Dst, f32>,
    {
        fn try_pool_to(self) -> Result<Tensor<Dst, f32, D, T>, Self::Err> {
            try_unary_op(Default::default(), self)
        }
    }
}

pub trait AvgPool2D<T>: HasErr {
    fn avg_pool2d<const K: usize, const S: usize, const P: usize>(self) -> T
    where
        Self: internals::PoolTo<T, unary_ops::pooling::Avg, K, S, P>,
    {
        self.try_pool_to().unwrap()
    }
    fn try_avg_pool2d<const K: usize, const S: usize, const P: usize>(self) -> Result<T, Self::Err>
    where
        Self: internals::PoolTo<T, unary_ops::pooling::Avg, K, S, P>,
    {
        self.try_pool_to()
    }
}

pub trait MinPool2D<T>: HasErr {
    fn min_pool2d<const K: usize, const S: usize, const P: usize>(self) -> T
    where
        Self: internals::PoolTo<T, unary_ops::pooling::Min, K, S, P>,
    {
        self.try_pool_to().unwrap()
    }
    fn try_min_pool2d<const K: usize, const S: usize, const P: usize>(self) -> Result<T, Self::Err>
    where
        Self: internals::PoolTo<T, unary_ops::pooling::Min, K, S, P>,
    {
        self.try_pool_to()
    }
}

pub trait MaxPool2D<T>: HasErr {
    fn max_pool2d<const K: usize, const S: usize, const P: usize>(self) -> T
    where
        Self: internals::PoolTo<T, unary_ops::pooling::Max, K, S, P>,
    {
        self.try_pool_to().unwrap()
    }
    fn try_max_pool2d<const K: usize, const S: usize, const P: usize>(self) -> Result<T, Self::Err>
    where
        Self: internals::PoolTo<T, unary_ops::pooling::Max, K, S, P>,
    {
        self.try_pool_to()
    }
}

impl<Src: HasErr, Dst> AvgPool2D<Dst> for Src {}
impl<Src: HasErr, Dst> MinPool2D<Dst> for Src {}
impl<Src: HasErr, Dst> MaxPool2D<Dst> for Src {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{AsArray, Randn};
    use crate::tensor::*;
    use crate::tensor_ops::impl_backward::TryBackward;
    use crate::tensor_ops::impl_mean::MeanTo;
    use crate::tensor_ops::impl_sum::SumTo;
    use crate::tensor_ops::map::TryExp;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_pool2d_3d_max2d_eq_grads() {
        let dev = build_test_device!();
        let x = dev.tensor([[[1.0f32, 1., 0.5, 0.2], [0.2, 0.2, 0.5, 1.2]]]);
        let r = x.trace().max_pool2d::<2, 1, 0>();
        assert_close(&r.as_array(), &[[[1., 1., 1.2]]]);
        let g = r.sum().backward();
        assert_close(
            &g.get(&x).as_array(),
            &[[[1., 2., 0., 0.], [0., 0., 0., 1.]]],
        );
    }

    #[test]
    fn test_pool2d_3d_min2d_eq_grads() {
        let dev = build_test_device!();
        let x = dev.tensor([[[1., 1., 0.5, 0.2], [0.2, 0.2, 0.5, 1.2]]]);
        let r = x.trace().min_pool2d::<2, 1, 0>();
        assert_close(&r.as_array(), &[[[0.2, 0.2, 0.2]]]);
        let g = r.sum().backward();
        assert_close(
            &g.get(&x).as_array(),
            &[[[0., 0., 0., 1.], [1., 2., 0., 0.]]],
        );
    }

    #[test]
    fn test_pool2d_3d_max2d() {
        let dev = build_test_device!(234);
        let x: Tensor3D<2, 3, 4, _> = dev.randn();
        let r = x.trace().max_pool2d::<2, 2, 0>();
        assert_close(
            &r.as_array(),
            &[[[1.79155397, 1.10126066]], [[1.14464748, 2.26301837]]],
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&x).as_array(),
            &[
                [[1.49969184, 0., 0., 0.75198889],[0., 0., 0., 0.],[0., 0., 0., 0.]],
                [[0., 0., 2.40301466, 0.],[0.78533345, 0., 0., 0.],[0., 0., 0., 0.]]
            ]
        );
    }

    #[test]
    fn test_pool2d_3d_min2d() {
        let dev = build_test_device!(234);
        let x: Tensor3D<2, 3, 4, _> = dev.randn();
        let r = x.trace().min_pool2d::<2, 2, 0>();
        assert_close(
            &r.as_array(),
            &[[[-1.09635627, -1.07717276]], [[-0.01996479, -1.82562149]]],
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&x).as_array(),
            &[
                [[0., 0., 0., 0.],[0.083521545, 0., 0., 0.08513925],[0., 0., 0., 0.]],
                [[0., 0.2450583, 0., 0.04027937],[0., 0., 0., 0.],[0., 0., 0., 0.]],
            ],
        );
    }

    #[test]
    fn test_pool2d_3d_avg2d() {
        let dev = build_test_device!(234);
        let x: Tensor3D<2, 3, 4, _> = dev.randn();
        let r = x.trace().avg_pool2d::<2, 2, 0>();
        // assert_close(
        //     &r.as_array(),
        //     &[[[0.03031558, -0.25052455]], [[0.39499030, 0.04878314]]],
        // );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&x).as_array(),
            &[
                [[0.06442373, 0.06442373, 0.048649523, 0.048649523],[0.06442373, 0.06442373, 0.048649523, 0.048649523],[0.0, 0.0, 0.0, 0.0]],
                [[0.09277311, 0.09277311, 0.06562454, 0.06562454],[0.09277311, 0.09277311, 0.06562454, 0.06562454],[0.0, 0.0, 0.0, 0.0]]
            ]
        );
    }

    #[test]
    fn test_pool2d_4d_avg2d() {
        let dev = build_test_device!(234);
        let x: Tensor4D<2, 4, 2, 2, _> = dev.randn();
        let r = x.trace().avg_pool2d::<1, 2, 0>();
        assert_close(
            &r.as_array(),
            &[
                [[[1.791554]], [[-1.0963563]], [[0.86268073]], [[0.28538525]]],
                [[[1.1446475]], [[0.2833436]], [[-1.2026008]], [[0.21327473]]],
            ],
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&x).as_array(),
            &[
                [[[0.7498459, 0.0], [0.0, 0.0]],[[0.041760772, 0.0], [0.0, 0.0]],[[0.29618803, 0.0], [0.0, 0.0]],[[0.16628431, 0.0], [0.0, 0.0]]],
                [[[0.39266673, 0.0], [0.0, 0.0]],[[0.16594516, 0.0], [0.0, 0.0]],[[0.037551485, 0.0], [0.0, 0.0]],[[0.15471558, 0.0], [0.0, 0.0]]]
            ]
        );
    }
}
