mod cpu_kernel;

use crate::{
    gradients::Tape,
    shapes::{Const, Dim, Dtype},
    tensor::{DeviceStorage, HasErr, PutTape, SplitTape, Tensor},
};

pub(super) mod pooling {
    pub struct Max;
    pub struct Min;
    pub struct Avg;
}

pub trait Pool2DKernel<E: Dtype, Kind, const K: usize, const S: usize, const P: usize>:
    DeviceStorage
{
    #[rustfmt::skip]
    fn forward<C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(C, Const<H>, Const<W>), E>,
    ) -> Result<
        Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), E>,
        Self::Err,
    >;

    #[rustfmt::skip]
    fn backward<C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(C, Const<H>, Const<W>), E>,
        grad_inp: &mut Self::Storage<(C, Const<H>, Const<W>), E>,
        out: &Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), E>,
        grad_out: &Self::Storage<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), E>,
    ) -> Result<(), Self::Err>;
}

pub trait Pool2DBatchedKernel<E: Dtype, Kind, const K: usize, const S: usize, const P: usize>:
    DeviceStorage
{
    #[rustfmt::skip]
    fn forward<B: Dim, C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(B, C, Const<H>, Const<W>), E>,
    ) -> Result<
        Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), E>,
        Self::Err,
    >;

    #[rustfmt::skip]
    fn backward<B: Dim, C: Dim, const H: usize, const W: usize>(
        &self,
        inp: &Self::Storage<(B, C, Const<H>, Const<W>), E>,
        grad_inp: &mut Self::Storage<(B, C, Const<H>, Const<W>), E>,
        out: &Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), E>,
        grad_out: &Self::Storage<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), E>,
    ) -> Result<(), Self::Err>;
}

pub trait TryPool2DTo<const K: usize, const S: usize, const P: usize>: HasErr {
    type Output;

    fn try_avg_pool2d_to(self) -> Result<Self::Output, Self::Err>;
    fn try_min_pool2d_to(self) -> Result<Self::Output, Self::Err>;
    fn try_max_pool2d_to(self) -> Result<Self::Output, Self::Err>;
}

pub trait TryPool2D {
    fn avg_pool2d<const K: usize, const S: usize, const P: usize>(self) -> Self::Output
    where
        Self: TryPool2DTo<K, S, P>,
    {
        self.try_avg_pool2d_to().unwrap()
    }
    fn try_avg_pool2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: TryPool2DTo<K, S, P>,
    {
        self.try_avg_pool2d_to()
    }

    fn min_pool2d<const K: usize, const S: usize, const P: usize>(self) -> Self::Output
    where
        Self: TryPool2DTo<K, S, P>,
    {
        self.try_min_pool2d_to().unwrap()
    }
    fn try_min_pool2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: TryPool2DTo<K, S, P>,
    {
        self.try_min_pool2d_to()
    }

    fn max_pool2d<const K: usize, const S: usize, const P: usize>(self) -> Self::Output
    where
        Self: TryPool2DTo<K, S, P>,
    {
        self.try_max_pool2d_to().unwrap()
    }
    fn try_max_pool2d<const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: TryPool2DTo<K, S, P>,
    {
        self.try_max_pool2d_to()
    }
}

impl<T> TryPool2D for T {}

impl<C: Dim, const H: usize, const W: usize, D: DeviceStorage, T: Tape<D>>
    Tensor<(C, Const<H>, Const<W>), f32, D, T>
{
    #[rustfmt::skip]
    fn try_pool2d<Kind: 'static, const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Result<Tensor<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32, D, T>, D::Err>
    where
        D: Pool2DKernel<f32, Kind, K, S, P>,
    {
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.upgrade(inp.device.forward::<C, H, W>(&inp.storage)?);
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device.backward::<C, H, W>(&inp.storage, grad_inp, &phantom_out.storage, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        C: Dim,
        const H: usize,
        const W: usize,
        D: DeviceStorage,
        T: Tape<D>,
    > TryPool2DTo<K, S, P> for Tensor<(C, Const<H>, Const<W>), f32, D, T>
where
    D: Pool2DKernel<f32, pooling::Avg, K, S, P>
        + Pool2DKernel<f32, pooling::Min, K, S, P>
        + Pool2DKernel<f32, pooling::Max, K, S, P>,
    (
        C,
        Const<{ (H + 2 * P - K) / S + 1 }>,
        Const<{ (W + 2 * P - K) / S + 1 }>,
    ): Sized,
{
    #[rustfmt::skip]
    type Output = Tensor<(C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32, D, T>;

    fn try_avg_pool2d_to(self) -> Result<Self::Output, Self::Err> {
        self.try_pool2d::<pooling::Avg, K, S, P>()
    }
    fn try_max_pool2d_to(self) -> Result<Self::Output, Self::Err> {
        self.try_pool2d::<pooling::Max, K, S, P>()
    }
    fn try_min_pool2d_to(self) -> Result<Self::Output, Self::Err> {
        self.try_pool2d::<pooling::Min, K, S, P>()
    }
}

impl<B: Dim, C: Dim, const H: usize, const W: usize, D: DeviceStorage, T: Tape<D>>
    Tensor<(B, C, Const<H>, Const<W>), f32, D, T>
{
    #[rustfmt::skip]
    fn try_pool2d<Kind: 'static, const K: usize, const S: usize, const P: usize>(
        self,
    ) -> Result<
        Tensor<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32, D, T>,
        D::Err,
    >
    where
        D: Pool2DBatchedKernel<f32, Kind, K, S, P>,
    {
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.upgrade(inp.device.forward::<B, C, H, W>(&inp.storage)?);
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device.backward::<B, C, H, W>(&inp.storage, grad_inp, &phantom_out.storage, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        B: Dim,
        C: Dim,
        const H: usize,
        const W: usize,
        D: DeviceStorage,
        T: Tape<D>,
    > TryPool2DTo<K, S, P> for Tensor<(B, C, Const<H>, Const<W>), f32, D, T>
where
    D: Pool2DBatchedKernel<f32, pooling::Avg, K, S, P>
        + Pool2DBatchedKernel<f32, pooling::Min, K, S, P>
        + Pool2DBatchedKernel<f32, pooling::Max, K, S, P>,
    (
        B,
        C,
        Const<{ (H + 2 * P - K) / S + 1 }>,
        Const<{ (W + 2 * P - K) / S + 1 }>,
    ): Sized,
{
    #[rustfmt::skip]
    type Output = Tensor<(B, C, Const<{ (H + 2 * P - K) / S + 1 }>, Const<{ (W + 2 * P - K) / S + 1 }>), f32, D, T>;

    fn try_avg_pool2d_to(self) -> Result<Self::Output, Self::Err> {
        self.try_pool2d::<pooling::Avg, K, S, P>()
    }
    fn try_max_pool2d_to(self) -> Result<Self::Output, Self::Err> {
        self.try_pool2d::<pooling::Max, K, S, P>()
    }
    fn try_min_pool2d_to(self) -> Result<Self::Output, Self::Err> {
        self.try_pool2d::<pooling::Min, K, S, P>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_pool2d_3d_max2d_eq_grads() {
        let dev = build_test_device!();
        let x = dev.tensor([[[1.0f32, 1., 0.5, 0.2], [0.2, 0.2, 0.5, 1.2]]]);
        let r = x.trace().max_pool2d::<2, 1, 0>();
        assert_close(&r.array(), &[[[1., 1., 1.2]]]);
        let g = r.sum().backward();
        assert_close(&g.get(&x).array(), &[[[1., 2., 0., 0.], [0., 0., 0., 1.]]]);
    }

    #[test]
    fn test_pool2d_3d_min2d_eq_grads() {
        let dev = build_test_device!();
        let x = dev.tensor([[[1., 1., 0.5, 0.2], [0.2, 0.2, 0.5, 1.2]]]);
        let r = x.trace().min_pool2d::<2, 1, 0>();
        assert_close(&r.array(), &[[[0.2, 0.2, 0.2]]]);
        let g = r.sum().backward();
        assert_close(&g.get(&x).array(), &[[[0., 0., 0., 1.], [1., 2., 0., 0.]]]);
    }

    #[test]
    fn test_pool2d_3d_max2d() {
        let dev = build_test_device!(234);
        let x: Tensor3D<2, 3, 4, _> = dev.randn();
        let r = x.trace().max_pool2d::<2, 2, 0>();
        assert_close(
            &r.array(),
            &[[[1.79155397, 1.10126066]], [[1.14464748, 2.26301837]]],
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&x).array(),
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
            &r.array(),
            &[[[-1.09635627, -1.07717276]], [[-0.01996479, -1.82562149]]],
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&x).array(),
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
        //     &r.array(),
        //     &[[[0.03031558, -0.25052455]], [[0.39499030, 0.04878314]]],
        // );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&x).array(),
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
            &r.array(),
            &[
                [[[1.791554]], [[-1.0963563]], [[0.86268073]], [[0.28538525]]],
                [[[1.1446475]], [[0.2833436]], [[-1.2026008]], [[0.21327473]]],
            ],
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close(
            &g.get(&x).array(),
            &[
                [[[0.7498459, 0.0], [0.0, 0.0]],[[0.041760772, 0.0], [0.0, 0.0]],[[0.29618803, 0.0], [0.0, 0.0]],[[0.16628431, 0.0], [0.0, 0.0]]],
                [[[0.39266673, 0.0], [0.0, 0.0]],[[0.16594516, 0.0], [0.0, 0.0]],[[0.037551485, 0.0], [0.0, 0.0]],[[0.15471558, 0.0], [0.0, 0.0]]]
            ]
        );
    }
}
