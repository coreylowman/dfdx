mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::*,
    tensor::{DeviceStorage, HasErr, PutTape, SplitTape, Tape, Tensor, ZerosTensor},
};

use super::conv2d::ConvAlgebra;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Pool2DOp {
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub batch: usize,
    pub chan: usize,
    pub h_in: usize,
    pub h_out: usize,
    pub w_in: usize,
    pub w_out: usize,
}

impl Pool2DOp {
    fn new(k: usize, s: usize, p: usize, [b, c, h_in, w_in]: [usize; 4]) -> Self {
        Self {
            kernel: k,
            stride: s,
            padding: p,
            batch: b,
            chan: c,
            h_in,
            h_out: (h_in + 2 * p - k) / s + 1,
            w_in,
            w_out: (w_in + 2 * p - k) / s + 1,
        }
    }
}

macro_rules! pool2d {
    (Kernel=$Kernel:ident, ConstTrait=$ConstTrait:ident, TryTrait=$TryTrait:ident, Meth=$Meth:ident, TryMeth=$TryMeth:ident) => {
        pub trait $Kernel<E: Unit>: DeviceStorage {
            fn forward<I: Shape, O: Shape>(
                &self,
                op: Pool2DOp,
                inp: &Tensor<I, E, Self>,
                out: &mut Tensor<O, E, Self>,
            ) -> Result<(), Self::Err>;

            fn backward<I: Shape, O: Shape>(
                &self,
                op: Pool2DOp,
                inp: &Tensor<I, E, Self>,
                grad_inp: &mut Self::Vec<E>,
                out: &Tensor<O, E, Self>,
                grad_out: &Self::Vec<E>,
            ) -> Result<(), Self::Err>;
        }

        pub trait $ConstTrait<const K: usize, const S: usize, const P: usize>: HasErr {
            type Output;
            fn try_pool2d(self) -> Result<Self::Output, Self::Err>;
        }

        pub trait $TryTrait {
            fn $Meth<const K: usize, const S: usize, const P: usize>(self) -> Self::Output
            where
                Self: $ConstTrait<K, S, P>,
            {
                self.try_pool2d().unwrap()
            }
            fn $TryMeth<const K: usize, const S: usize, const P: usize>(
                self,
            ) -> Result<Self::Output, Self::Err>
            where
                Self: $ConstTrait<K, S, P>,
            {
                self.try_pool2d()
            }
        }
        impl<S: Shape, E: Dtype, D: DeviceStorage, T> $TryTrait for Tensor<S, E, D, T> {}

        impl<
                C: Dim,
                H: Dim + ConvAlgebra<K, S, P>,
                W: Dim + ConvAlgebra<K, S, P>,
                E: Dtype,
                D: $Kernel<E> + ZerosTensor<E>,
                T: 'static + Tape<E, D>,
                const K: usize,
                const S: usize,
                const P: usize,
            > $ConstTrait<K, S, P> for Tensor<(C, H, W), E, D, T>
        {
            type Output = Tensor<(C, H::Convolved, W::Convolved), E, D, T>;

            fn try_pool2d(self) -> Result<Self::Output, Self::Err> {
                let h = self.shape.1;
                let w = self.shape.2;

                let &(chan, _, _) = self.shape();
                let op = Pool2DOp::new(K, S, P, [1, chan.size(), h.size(), w.size()]);
                let (inp, mut tape) = self.split_tape();
                let mut out =
                    inp.device
                        .try_zeros_like(&(chan, h.convolve_dim(), w.convolve_dim()))?;
                inp.device.forward(op, &inp, &mut out)?;
                let inp_ghost = inp.ghost();
                let out_ghost = out.ghost();
                let out_clone = out.clone();
                tape.add_backward_op(move |grads| {
                    grads.try_alloc_for(&inp_ghost)?;
                    grads.try_alloc_for(&out_ghost)?;
                    let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
                    inp_ghost
                        .dev
                        .backward(op, &inp, grad_inp, &out_clone, grad_out)
                });
                Ok(out.put_tape(tape))
            }
        }

        impl<
                B: Dim,
                C: Dim,
                H: Dim + ConvAlgebra<K, S, P>,
                W: Dim + ConvAlgebra<K, S, P>,
                E: Dtype,
                D: $Kernel<E> + ZerosTensor<E>,
                T: 'static + Tape<E, D>,
                const K: usize,
                const S: usize,
                const P: usize,
            > $ConstTrait<K, S, P> for Tensor<(B, C, H, W), E, D, T>
        {
            type Output = Tensor<(B, C, H::Convolved, W::Convolved), E, D, T>;

            fn try_pool2d(self) -> Result<Self::Output, Self::Err> {
                let h = self.shape.2;
                let w = self.shape.3;

                let &(batch, chan, _, _) = self.shape();
                let op = Pool2DOp::new(K, S, P, [batch.size(), chan.size(), h.size(), w.size()]);
                let (inp, mut tape) = self.split_tape();
                let mut out = inp.device.try_zeros_like(&(
                    batch,
                    chan,
                    h.convolve_dim(),
                    w.convolve_dim(),
                ))?;
                inp.device.forward(op, &inp, &mut out)?;
                let inp_ghost = inp.ghost();
                let out_ghost = out.ghost();
                let out_clone = out.clone();
                tape.add_backward_op(move |grads| {
                    grads.try_alloc_for(&inp_ghost)?;
                    grads.try_alloc_for(&out_ghost)?;
                    let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
                    inp_ghost
                        .dev
                        .backward(op, &inp, grad_inp, &out_clone, grad_out)
                });
                Ok(out.put_tape(tape))
            }
        }
    };
}

pool2d!(
    Kernel = AvgPool2DKernel,
    ConstTrait = ConstAvgPool2D,
    TryTrait = TryAvgPool2D,
    Meth = avg_pool2d,
    TryMeth = try_avg_pool2d
);

pool2d!(
    Kernel = MaxPool2DKernel,
    ConstTrait = ConstMaxPool2D,
    TryTrait = TryMaxPool2D,
    Meth = max_pool2d,
    TryMeth = try_max_pool2d
);

pool2d!(
    Kernel = MinPool2DKernel,
    ConstTrait = ConstMinPool2D,
    TryTrait = TryMinPool2D,
    Meth = min_pool2d,
    TryMeth = try_min_pool2d
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_pool2d_3d_max2d_eq_grads() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([[[1.0, 1., 0.5, 0.2], [0.2, 0.2, 0.5, 1.2]]]);
        let r = x.leaky_trace().max_pool2d::<2, 1, 0>();
        assert_close_to_literal!(r, [[[1., 1., 1.2]]]);
        let g = r.sum().backward();
        assert_close_to_literal!(g.get(&x), [[[1., 2., 0., 0.], [0., 0., 0., 1.]]]);
    }

    #[test]
    fn test_pool2d_3d_min2d_eq_grads() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([[[1., 1., 0.5, 0.2], [0.2, 0.2, 0.5, 1.2]]]);
        let r = x.leaky_trace().min_pool2d::<2, 1, 0>();
        assert_close_to_literal!(r, [[[0.2, 0.2, 0.2]]]);
        let g = r.sum().backward();
        assert_close_to_literal!(g.get(&x), [[[0., 0., 0., 1.], [1., 2., 0., 0.]]]);
    }

    #[test]
    fn test_pool2d_3d_max2d() {
        let dev = TestDevice::seed_from_u64(234);
        let x: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let r = x.leaky_trace().max_pool2d::<2, 2, 0>();
        assert_close_to_literal!(r, [[[1.79155397, 1.10126066]], [[1.14464748, 2.26301837]]]);
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[1.49969184, 0., 0., 0.75198889],[0., 0., 0., 0.],[0., 0., 0., 0.]],
                [[0., 0., 2.40301466, 0.],[0.78533345, 0., 0., 0.],[0., 0., 0., 0.]]
            ]
        );
    }

    #[test]
    fn test_pool2d_3d_min2d() {
        let dev = TestDevice::seed_from_u64(234);
        let x: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let r = x.leaky_trace().min_pool2d::<2, 2, 0>();
        assert_close_to_literal!(
            r,
            [[[-1.09635627, -1.07717276]], [[-0.01996479, -1.82562149]]]
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[0., 0., 0., 0.],[0.083521545, 0., 0., 0.08513925],[0., 0., 0., 0.]],
                [[0., 0.2450583, 0., 0.04027937],[0., 0., 0., 0.],[0., 0., 0., 0.]],
            ]
        );
    }

    #[test]
    fn test_pool2d_3d_avg2d() {
        let dev = TestDevice::seed_from_u64(234);
        let x: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let r = x.leaky_trace().avg_pool2d::<2, 2, 0>();
        assert_close_to_literal!(r, [[[0.03031558, -0.25052455]], [[0.39499030, 0.04878314]]]);
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[0.06442373, 0.06442373, 0.048649523, 0.048649523],[0.06442373, 0.06442373, 0.048649523, 0.048649523],[0.0, 0.0, 0.0, 0.0]],
                [[0.09277311, 0.09277311, 0.06562454, 0.06562454],[0.09277311, 0.09277311, 0.06562454, 0.06562454],[0.0, 0.0, 0.0, 0.0]]
            ]
        );
    }

    #[test]
    fn test_pool2d_4d_avg2d() {
        let dev = TestDevice::seed_from_u64(234);
        let x: Tensor<Rank4<2, 4, 2, 2>, TestDtype, _> = dev.sample_normal();
        let r = x.leaky_trace().avg_pool2d::<1, 2, 0>();
        assert_close_to_literal!(
            r,
            [
                [[[1.791554]], [[-1.0963563]], [[0.86268073]], [[0.28538525]]],
                [[[1.1446475]], [[0.2833436]], [[-1.2026008]], [[0.21327473]]],
            ]
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[[0.7498459, 0.0], [0.0, 0.0]],[[0.041760772, 0.0], [0.0, 0.0]],[[0.29618803, 0.0], [0.0, 0.0]],[[0.16628431, 0.0], [0.0, 0.0]]],
                [[[0.39266673, 0.0], [0.0, 0.0]],[[0.16594516, 0.0], [0.0, 0.0]],[[0.037551485, 0.0], [0.0, 0.0]],[[0.15471558, 0.0], [0.0, 0.0]]]
            ]
        );
    }
}
