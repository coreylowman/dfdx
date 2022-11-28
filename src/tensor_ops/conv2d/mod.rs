mod cpu_kernel;

use crate::{
    arrays::{Const, Dim, Dtype, Rank3, Rank4},
    devices::DeviceStorage,
    gradients::Tape,
    tensor::{make_tensor, Tensor},
};

pub trait Conv2DKernel<
    E: Dtype,
    const C: usize,
    const O: usize,
    const K: usize,
    const S: usize,
    const P: usize,
>: DeviceStorage
{
    fn forward<const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<Rank3<C, H, W>, E>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, E>,
    ) -> Result<
        Self::Storage<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, E>,
        Self::Err,
    >;

    fn backward<const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<Rank3<C, H, W>, E>,
        grad_lhs: &mut Self::Storage<Rank3<C, H, W>, E>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, E>,
        grad_rhs: &mut Self::Storage<Rank4<O, C, K, K>, E>,
        grad_out: &Self::Storage<
            Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            E,
        >,
    ) -> Result<(), Self::Err>;
}

pub trait Conv2DBatchedKernel<
    E: Dtype,
    const C: usize,
    const O: usize,
    const K: usize,
    const S: usize,
    const P: usize,
>: DeviceStorage
{
    fn forward<B: Dim, const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<(B, Const<C>, Const<H>, Const<W>), E>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, E>,
    ) -> Result<
        Self::Storage<
            (
                B,
                Const<O>,
                Const<{ (H + 2 * P - K) / S + 1 }>,
                Const<{ (W + 2 * P - K) / S + 1 }>,
            ),
            E,
        >,
        Self::Err,
    >;

    fn backward<B: Dim, const H: usize, const W: usize>(
        &self,
        lhs: &Self::Storage<(B, Const<C>, Const<H>, Const<W>), E>,
        grad_lhs: &mut Self::Storage<(B, Const<C>, Const<H>, Const<W>), E>,
        rhs: &Self::Storage<Rank4<O, C, K, K>, E>,
        grad_rhs: &mut Self::Storage<Rank4<O, C, K, K>, E>,
        grad_out: &Self::Storage<
            (
                B,
                Const<O>,
                Const<{ (H + 2 * P - K) / S + 1 }>,
                Const<{ (W + 2 * P - K) / S + 1 }>,
            ),
            E,
        >,
    ) -> Result<(), Self::Err>;
}

impl<const C: usize, const H: usize, const W: usize, D: DeviceStorage, T: Tape<D>>
    Tensor<Rank3<C, H, W>, f32, D, T>
{
    /// **Requires Nightly** Perform a 2d convolution
    ///
    /// TODO docstring
    pub fn conv2d<const O: usize, const K: usize, const S: usize, const P: usize>(
        self,
        filters: Tensor<Rank4<O, C, K, K>, f32, D>,
    ) -> Tensor<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32, D, T>
    where
        D: Conv2DKernel<f32, C, O, K, S, P>,
    {
        self.try_conv2d(filters).unwrap()
    }

    /// **Requires Nightly** Fallible version of 3d conv2d
    pub fn try_conv2d<const O: usize, const K: usize, const S: usize, const P: usize>(
        self,
        filters: Tensor<Rank4<O, C, K, K>, f32, D>,
    ) -> Result<
        Tensor<Rank3<O, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32, D, T>,
        D::Err,
    >
    where
        D: Conv2DKernel<f32, C, O, K, S, P>,
    {
        let (lhs, ltape) = self.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut tape = ltape.merge(rtape);
        let storage = lhs.device.forward(&lhs.storage, &rhs.storage)?;
        let out = make_tensor(&lhs.device, storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out)?;
            lhs.device
                .backward(&lhs.storage, grad_lhs, &rhs.storage, grad_rhs, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

impl<B: Dim, const C: usize, const H: usize, const W: usize, D: DeviceStorage, T: Tape<D>>
    Tensor<(B, Const<C>, Const<H>, Const<W>), f32, D, T>
{
    /// **Requires Nightly** Perform a batched 2d convolution
    ///
    /// TODO docstring
    pub fn conv2d<const O: usize, const K: usize, const S: usize, const P: usize>(
        self,
        filters: Tensor<Rank4<O, C, K, K>, f32, D>,
    ) -> Tensor<
        (
            B,
            Const<O>,
            Const<{ (H + 2 * P - K) / S + 1 }>,
            Const<{ (W + 2 * P - K) / S + 1 }>,
        ),
        f32,
        D,
        T,
    >
    where
        D: Conv2DBatchedKernel<f32, C, O, K, S, P>,
    {
        self.try_conv2d(filters).unwrap()
    }

    /// **Requires Nightly** Fallible batched 2d convolution
    ///
    /// TODO docstring
    pub fn try_conv2d<const O: usize, const K: usize, const S: usize, const P: usize>(
        self,
        filters: Tensor<Rank4<O, C, K, K>, f32, D>,
    ) -> Result<
        Tensor<
            (
                B,
                Const<O>,
                Const<{ (H + 2 * P - K) / S + 1 }>,
                Const<{ (W + 2 * P - K) / S + 1 }>,
            ),
            f32,
            D,
            T,
        >,
        D::Err,
    >
    where
        D: Conv2DBatchedKernel<f32, C, O, K, S, P>,
    {
        let (lhs, ltape) = self.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut tape = ltape.merge(rtape);
        let storage = lhs.device.forward(&lhs.storage, &rhs.storage)?;
        let out = make_tensor(&lhs.device, storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out)?;
            lhs.device
                .backward(&lhs.storage, grad_lhs, &rhs.storage, grad_rhs, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arrays::{Axes2, Axes3},
        devices::{AsArray, Randn},
        tensor::*,
        tensor_ops::*,
        tests::{assert_close, build_test_device, AssertClose},
    };

    #[test]
    /// Produced by
    /// ```python
    /// q = torch.nn.Conv2d(1, 2, 2)
    /// x = torch.randn(1, 2, 3, requires_grad=True)
    /// q(x).exp().mean().backward()
    /// ```
    fn test_conv2d_default_stride_and_padding() {
        let dev = build_test_device!();
        let weight = dev.tensor([
            [[[-0.04958433, -0.43007267], [0.01935136, 0.09778714]]],
            [[[0.44083858, -0.20507240], [-0.30017477, -0.10937047]]],
        ]);
        let bias = dev.tensor([0.36406237, -0.30981010]);
        let x = dev.tensor([[
            [-0.86713916, 0.52773184, -0.95238322],
            [-0.64531374, 0.77809018, -0.49099201],
        ]]);
        let result = x.trace().conv2d::<2, 2, 1, 0>(weight.clone())
            + BroadcastTo::<_, Axes2<1, 2>>::broadcast(bias.trace());
        assert_close(
            &result.as_array(),
            &[[[0.24369538, 0.71453357]], [[-0.69169492, -0.06172103]]],
        );
        let g = result.exp().mean().backward();
        assert_close(
            &g.get(&x).as_array(),
            &[[
                [0.03936806, -0.08457474, -0.26788417],
                [-0.03140351, -0.04316529, 0.02424446],
            ]],
        );
        assert_close(
            &g.get(&weight).as_array(),
            &[
                [[[-0.00703794, -0.31814471], [0.19160703, -0.00260070]]],
                [[[0.01548620, -0.15778227], [0.10209797, -0.01799832]]],
            ],
        );
        assert_close(&g.get(&bias).as_array(), &[0.82979727, 0.36021793]);
    }

    #[test]
    /// Produced by
    /// ```python
    /// q = torch.nn.Conv2d(1, 2, 2, stride=2)
    /// x = torch.randn(1, 2, 3, requires_grad=True)
    /// q(x).exp().mean().backward()
    /// ```
    fn test_conv2d_stride_2() {
        let dev = build_test_device!();
        let weight = dev.tensor([
            [[[0.44704646, -0.29563826], [0.29228759, -0.16575140]]],
            [[[-0.30488998, 0.25222939], [0.13279295, 0.38153177]]],
        ]);

        let bias = dev.tensor([-0.44699109, 0.38371694]);

        let x = dev.tensor([[
            [0.37100124, -0.59504986, -1.19781005],
            [-0.31547278, 0.58071911, 0.86612970],
        ]]);

        let result = x.trace().conv2d::<2, 2, 2, 0>(weight.clone())
            + BroadcastTo::<_, Axes2<1, 2>>::broadcast(bias.trace());
        assert_close(&result.as_array(), &[[[-0.29368058]], [[0.30018353]]]);

        let g = result.exp().mean().backward();

        assert_close(
            &g.get(&x).as_array(),
            &[[[-0.03917716, 0.06006697, 0.], [0.19859464, 0.19576924, 0.]]],
        );

        assert_close(
            &g.get(&weight).as_array(),
            &[
                [[[0.13829342, -0.22180916], [-0.11759478, 0.21646728]]],
                [[[0.25044560, -0.40169036], [-0.21296094, 0.39201635]]],
            ],
        );

        assert_close(&g.get(&bias).as_array(), &[0.37275729, 0.67505330]);
    }

    #[test]
    fn test_conv2d_padding_1() {
        let dev = build_test_device!();
        #[rustfmt::skip]
            let weight = dev.tensor([
                [[[0.10215953, 0.06263646], [-0.04124039, -0.09729567]], [[-0.32656857, 0.24254093], [-0.27209827, 0.15361503]]],
                [[[0.03449896, 0.22931078], [-0.17652659, 0.08222872]],[[-0.06016779, 0.29082409], [-0.19154115, 0.13483226]]],
                [[[-0.14262493, 0.19654515], [0.15921101, 0.01759464]],[[0.16749159, 0.33096817], [0.28376505, -0.05524009]]],
            ]);

        let bias = dev.tensor([-0.22854491, 0.28763595, 0.20709404]);

        let x = dev.tensor([[[-0.32224107, -0.32800716]], [[-1.13570976, 0.93713200]]]);

        let result = x.trace().conv2d::<3, 2, 1, 1>(weight.clone())
            + BroadcastTo::<_, Axes2<1, 2>>::broadcast(bias.trace());

        #[rustfmt::skip]
            assert_close(
                &result.as_array(),
                &[
                    [[-0.37165433, 0.26964033, -0.47000977],[-0.52418506, 0.3161699, -0.56809187]],
                    [[0.10800815, 0.66143924, 0.16603859],[-0.11654915, 0.5421771, 0.21993488]],
                    [[0.26416105, -0.22402346, 0.420797],[-0.23212466, 0.3085245, 0.41083777]],
                ],
            );

        let g = result.exp().mean().backward();

        assert_close(
            &g.get(&x).as_array(),
            &[[[0.010052743, 0.038219165]], [[0.0013861917, 0.096129306]]],
        );

        #[rustfmt::skip]
            assert_close(
                &g.get(&weight).as_array(),
                &[
                    [[[-0.03488452, -0.035597768], [-0.03483199, -0.036207683]],[[-0.05705857, 0.03406856], [-0.05008337, 0.024666183]]],
                    [[[-0.053492695, -0.04727108], [-0.05620105, -0.055251926]],[[-0.04363727, 0.033381317], [-0.0607851, 0.030584559]]],
                    [[[-0.051853612, -0.03900232], [-0.04206547, -0.037880093]],[[-0.0073834136, 0.0208545], [0.02886929, -0.040557314]]],
                ],
            );

        assert_close(
            &g.get(&bias).as_array(),
            &[0.28636602, 0.44933242, 0.40484178],
        );
    }

    #[test]
    fn test_conv2d_stride_3_padding_4() {
        let dev = build_test_device!();
        #[rustfmt::skip]
            let weight = dev.tensor([
                [[[-0.10252278, -0.14387409, -0.14627469],[0.28396228, -0.14590892, 0.29269591],[0.01090384, 0.14785287, 0.29242596]]],
                [[[-0.31163597, 0.13224581, -0.20954299],[0.27902845, -0.14735751, 0.14001134],[-0.05224654, 0.16499066, -0.13981307]]],
            ]);

        let bias = dev.tensor([-0.07123789, -0.17244765]);

        #[rustfmt::skip]
            let x = dev.tensor([[[0.69103152, 0.25624934],[-0.38448590, 0.03110456],[0.83753252, 0.53786588],[1.15540242, -0.54148245]]]);

        let result = x.trace().conv2d::<2, 3, 3, 4>(weight.clone())
            + BroadcastTo::<_, Axes2<1, 2>>::broadcast(bias.trace());

        #[rustfmt::skip]
            assert_close(
                &result.as_array(),
                &[
                    [[-0.07123789, -0.07123789, -0.07123789],[-0.07123789, -0.14481398, -0.07123789],[-0.07123789, -0.59748650, -0.07123789],[-0.07123789, -0.07123789, -0.07123789]],
                    [[-0.17244765, -0.17244765, -0.17244765],[-0.17244765, -0.3061839, -0.17244765],[-0.17244765, -0.42046443, -0.17244765],[-0.17244765, -0.17244765, -0.17244765]],
                ],
            );

        let g = result.exp().mean().backward();

        #[rustfmt::skip]
            assert_close(
                &g.get(&x).as_array(),
                &[[[-0.009780421, 0.01484663],[0.010391434, 0.0062526874],[0.00032053515, -0.009087289],[-0.0073772445, 0.0105412705]]],
            );

        #[rustfmt::skip]
            assert_close(
                &g.get(&weight).as_array(),
                &[
                    [[[0.0, 0.019200183, 0.012330416],[0.0, 0.051398464, -0.003175714],[0.0, -0.013860448, 0.0011212977]]],
                    [[[0.0, 0.02291844, 0.01471829],[0.0, 0.05281557, -0.0069562597],[0.0, -0.011794927, 0.00095419877]]],
                ],
            );

        assert_close(&g.get(&bias).as_array(), &[0.44699076, 0.408709]);
    }

    #[test]
    fn test_batched_conv2d() {
        let dev = build_test_device!();
        let weight = dev.tensor([
            [[[0.05998272]], [[-0.07759511]]],
            [[[0.68307382]], [[-0.56570816]]],
            [[[0.31137520]], [[0.41600472]]],
        ]);

        let bias = dev.tensor([0.49647599, 0.15591705, -0.12342280]);

        #[rustfmt::skip]
            let x = dev.tensor([
                [[[-0.5396145, -2.43986344], [-0.01883135, -1.19915044]],[[-1.30589044, 2.05276346], [-0.20004864, 0.19919693]]],
                [[[-0.22305037, 0.63030297], [0.65323567, -0.68972057]],[[-0.50617385, -0.87281805], [0.30253950, -1.75082350]]],
                [[[1.65487242, 0.44441956], [-0.45107457, 1.41857898]],[[1.00477660, -0.16381662], [0.40009478, -0.57880658]]],
            ]);
        let result = x.trace().conv2d::<3, 1, 1, 0>(weight.clone())
            + BroadcastTo::<_, Axes3<0, 2, 3>>::broadcast(bias.trace());

        #[rustfmt::skip]
        result.as_array().assert_close(
            &[
                [[[0.56543916, 0.19084194], [0.51086920, 0.40909100]],[[0.52607340, -2.67195487], [0.25622299, -0.77587855]],[[-0.83470196, -0.02917651], [-0.21250761, -0.41394162]]],
                [[[0.52237344, 0.60200971], [0.51218325, 0.59096003]],[[0.28990385, 1.08022082], [0.43097615, 0.67524213]],[[-0.40344587, -0.29025853], [0.20583645, -1.06653547]]],
                [[[0.51777399, 0.53584486], [0.43837392, 0.62647879]],[[0.71790677, 0.55216080], [-0.37853706, 1.45234680]],[[0.80985522, -0.05319006], [-0.09743492, 0.07750125]]],
            ],
            1e-4,
        );
        let g = result.exp().mean().backward();

        #[rustfmt::skip]
        g.get(&x).as_array().assert_close(
            &[
                [[[0.03879637, 0.01172858], [0.03428607, 0.01695974]],[[-0.02537140, 0.00752865], [-0.01455240, -0.00283930]]],
                [[[0.03394239, 0.06539788], [0.04260371, 0.04326087]],[[-0.01691348, -0.04157412], [-0.01358072, -0.03078515]]],
                [[[0.06113625, 0.04400696], [0.02342399, 0.09354331]],[[-0.00986115, -0.02002173], [-0.00362044, -0.05869440]]],
            ],
            1e-4,
        );

        #[rustfmt::skip]
        assert_close(
            &g.get(&weight).as_array(),
            &[[[[0.01032944]], [[-0.11132676]]],[[[0.26300028]], [[-0.24666277]]],[[[0.07612189]], [[0.05598290]]]],
        );

        assert_close(
            &g.get(&bias).as_array(),
            &[0.55381978, 0.55677116, 0.30686682],
        );
    }

    #[test]
    fn test_conv2d_s4p3k2() {
        let dev = build_test_device!(432);

        let weight: Tensor4D<3, 5, 2, 2, _> = dev.randn();
        let bias: Tensor1D<3, _> = dev.randn();
        let x: Tensor3D<5, 7, 6, _> = dev.randn();

        let out = x.conv2d::<3, 2, 4, 3>(weight);
        let out = out + BroadcastTo::<_, Axes2<1, 2>>::broadcast(bias);

        #[rustfmt::skip]
        assert_close(&out.as_array(), &[
            [[-0.57176435, -0.57176435, -0.57176435],[-0.57176435, 1.0759051, 1.4307989],[-0.57176435, -0.86296344, -1.8794353]],
            [[0.29306656, 0.29306656, 0.29306656],[0.29306656, 0.9771965, 1.467767],[0.29306656, -6.367015, -2.3370528]],
            [[-0.19717735, -0.19717735, -0.19717735],[-0.19717735, 1.3412137, 2.9476144],[-0.19717735, 4.247249, -2.1779637]],
        ]);
    }
}
