use super::*;
use crate::{tensor_ops::*, tests::*};

#[test]
/// Produced by
/// ```python
/// q = torch.nn.Conv2d(1, 2, 2)
/// x = torch.sample_normal(1, 2, 3, requires_grad=True)
/// q(x).exp().mean().backward()
/// ```
fn test_conv2d_default_stride_and_padding() {
    let dev: TestDevice = Default::default();
    let weight = dev
        .tensor([
            [[[-0.04958433, -0.43007267], [0.01935136, 0.09778714]]],
            [[[0.44083858, -0.20507240], [-0.30017477, -0.10937047]]],
        ])
        .to_dtype::<TestDtype>();
    let bias = dev
        .tensor([0.36406237, -0.30981010])
        .to_dtype::<TestDtype>();
    let x = dev
        .tensor([[
            [-0.86713916, 0.52773184, -0.95238322],
            [-0.64531374, 0.77809018, -0.49099201],
        ]])
        .to_dtype::<TestDtype>();
    let result = (x.leaky_trace(), weight.clone())
        .conv2d(Const::<1>, Const::<0>, Const::<1>, Const::<1>)
        + bias.leaky_trace().broadcast::<_, Axes2<1, 2>>();
    assert_close_to_literal!(
        result,
        [[[0.24369538, 0.71453357]], [[-0.69169492, -0.06172103]]]
    );
    let g = result.exp().mean().backward();
    assert_close_to_literal!(
        g.get(&x),
        [[
            [0.03936806, -0.08457474, -0.26788417],
            [-0.03140351, -0.04316529, 0.02424446],
        ]]
    );
    assert_close_to_literal!(
        g.get(&weight),
        [
            [[[-0.00703794, -0.31814471], [0.19160703, -0.00260070]]],
            [[[0.01548620, -0.15778227], [0.10209797, -0.01799832]]],
        ]
    );
    assert_close_to_literal!(g.get(&bias), [0.82979727, 0.36021793]);
}

#[test]
/// Produced by
/// ```python
/// q = torch.nn.Conv2d(1, 2, 2, stride=2)
/// x = torch.sample_normal(1, 2, 3, requires_grad=True)
/// q(x).exp().mean().backward()
/// ```
fn test_conv2d_stride_2() {
    let dev: TestDevice = Default::default();
    let weight = dev
        .tensor([
            [[[0.44704646, -0.29563826], [0.29228759, -0.16575140]]],
            [[[-0.30488998, 0.25222939], [0.13279295, 0.38153177]]],
        ])
        .to_dtype::<TestDtype>();
    let bias = dev
        .tensor([-0.44699109, 0.38371694])
        .to_dtype::<TestDtype>();
    let x = dev
        .tensor([[
            [0.37100124, -0.59504986, -1.19781005],
            [-0.31547278, 0.58071911, 0.86612970],
        ]])
        .to_dtype::<TestDtype>();

    let result = (x.leaky_trace(), weight.clone())
        .conv2d(Const::<2>, Const::<0>, Const::<1>, Const::<1>)
        + bias.leaky_trace().broadcast::<_, Axes2<1, 2>>();
    assert_close_to_literal!(result, [[[-0.29368058]], [[0.30018353]]]);

    let g = result.exp().mean().backward();

    assert_close_to_literal!(
        g.get(&x),
        [[[-0.03917716, 0.06006697, 0.], [0.19859464, 0.19576924, 0.]]]
    );

    assert_close_to_literal!(
        g.get(&weight),
        [
            [[[0.13829342, -0.22180916], [-0.11759478, 0.21646728]]],
            [[[0.25044560, -0.40169036], [-0.21296094, 0.39201635]]],
        ]
    );

    assert_close_to_literal!(g.get(&bias), [0.37275729, 0.67505330]);
}

#[test]
fn test_conv2d_padding_1() {
    let dev: TestDevice = Default::default();
    #[rustfmt::skip]
        let weight = dev.tensor([
            [[[0.10215953, 0.06263646], [-0.04124039, -0.09729567]], [[-0.32656857, 0.24254093], [-0.27209827, 0.15361503]]],
            [[[0.03449896, 0.22931078], [-0.17652659, 0.08222872]],[[-0.06016779, 0.29082409], [-0.19154115, 0.13483226]]],
            [[[-0.14262493, 0.19654515], [0.15921101, 0.01759464]],[[0.16749159, 0.33096817], [0.28376505, -0.05524009]]],
        ]).to_dtype::<TestDtype>();
    let bias = dev
        .tensor([-0.22854491, 0.28763595, 0.20709404])
        .to_dtype::<TestDtype>();
    let x = dev
        .tensor([[[-0.32224107, -0.32800716]], [[-1.13570976, 0.93713200]]])
        .to_dtype::<TestDtype>();

    let result = (x.leaky_trace(), weight.clone())
        .conv2d(Const::<1>, Const::<1>, Const::<1>, Const::<1>)
        + bias.leaky_trace().broadcast::<_, Axes2<1, 2>>();

    #[rustfmt::skip]
        assert_close_to_literal!(
            result,
            [
                [[-0.37165433, 0.26964033, -0.47000977],[-0.52418506, 0.3161699, -0.56809187]],
                [[0.10800815, 0.66143924, 0.16603859],[-0.11654915, 0.5421771, 0.21993488]],
                [[0.26416105, -0.22402346, 0.420797],[-0.23212466, 0.3085245, 0.41083777]],
            ]
        );

    let g = result.exp().mean().backward();

    assert_close_to_literal!(
        g.get(&x),
        [[[0.010052743, 0.038219165]], [[0.0013861917, 0.096129306]]]
    );

    #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&weight),
            [
                [[[-0.03488452, -0.035597768], [-0.03483199, -0.036207683]],[[-0.05705857, 0.03406856], [-0.05008337, 0.024666183]]],
                [[[-0.053492695, -0.04727108], [-0.05620105, -0.055251926]],[[-0.04363727, 0.033381317], [-0.0607851, 0.030584559]]],
                [[[-0.051853612, -0.03900232], [-0.04206547, -0.037880093]],[[-0.0073834136, 0.0208545], [0.02886929, -0.040557314]]],
            ]
        );

    assert_close_to_literal!(g.get(&bias), [0.28636602, 0.44933242, 0.40484178]);
}

#[test]
fn test_conv2d_stride_3_padding_4() {
    let dev: TestDevice = Default::default();
    #[rustfmt::skip]
        let weight = dev.tensor([
            [[[-0.10252278, -0.14387409, -0.14627469],[0.28396228, -0.14590892, 0.29269591],[0.01090384, 0.14785287, 0.29242596]]],
            [[[-0.31163597, 0.13224581, -0.20954299],[0.27902845, -0.14735751, 0.14001134],[-0.05224654, 0.16499066, -0.13981307]]],
        ]).to_dtype::<TestDtype>();
    let bias = dev
        .tensor([-0.07123789, -0.17244765])
        .to_dtype::<TestDtype>();
    #[rustfmt::skip]
        let x = dev.tensor([[[0.69103152, 0.25624934],[-0.38448590, 0.03110456],[0.83753252, 0.53786588],[1.15540242, -0.54148245]]]).to_dtype::<TestDtype>();

    let result = (x.leaky_trace(), weight.clone())
        .conv2d(Const::<3>, Const::<4>, Const::<1>, Const::<1>)
        + bias.leaky_trace().broadcast::<_, Axes2<1, 2>>();

    #[rustfmt::skip]
        assert_close_to_literal!(
            result,
            [
                [[-0.07123789, -0.07123789, -0.07123789],[-0.07123789, -0.14481398, -0.07123789],[-0.07123789, -0.59748650, -0.07123789],[-0.07123789, -0.07123789, -0.07123789]],
                [[-0.17244765, -0.17244765, -0.17244765],[-0.17244765, -0.3061839, -0.17244765],[-0.17244765, -0.42046443, -0.17244765],[-0.17244765, -0.17244765, -0.17244765]],
            ]
        );

    let g = result.exp().mean().backward();

    #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [[[-0.009780421, 0.01484663],[0.010391434, 0.0062526874],[0.00032053515, -0.009087289],[-0.0073772445, 0.0105412705]]]
        );

    #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&weight),
            [
                [[[0.0, 0.019200183, 0.012330416],[0.0, 0.051398464, -0.003175714],[0.0, -0.013860448, 0.0011212977]]],
                [[[0.0, 0.02291844, 0.01471829],[0.0, 0.05281557, -0.0069562597],[0.0, -0.011794927, 0.00095419877]]],
            ]
        );

    assert_close_to_literal!(g.get(&bias), [0.44699076, 0.408709]);
}

#[test]
fn test_conv2d_s4p3k2() {
    let dev = TestDevice::seed_from_u64(432);

    let weight: Tensor<Rank4<3, 5, 2, 2>, TestDtype, _> = dev.sample_normal();
    let bias: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
    let x: Tensor<Rank3<5, 7, 6>, TestDtype, _> = dev.sample_normal();

    let out =
        (x.leaky_trace(), weight.clone()).conv2d(Const::<4>, Const::<3>, Const::<1>, Const::<1>);
    let out = out + bias.broadcast::<_, Axes2<1, 2>>();

    #[rustfmt::skip]
        assert_close_to_literal!(out, [
            [[-0.57176435, -0.57176435, -0.57176435],[-0.57176435, 1.0759051, 1.4307989],[-0.57176435, -0.86296344, -1.8794353]],
            [[0.29306656, 0.29306656, 0.29306656],[0.29306656, 0.9771965, 1.467767],[0.29306656, -6.367015, -2.3370528]],
            [[-0.19717735, -0.19717735, -0.19717735],[-0.19717735, 1.3412137, 2.9476144],[-0.19717735, 4.247249, -2.1779637]],
        ]);
}

#[test]
fn test_batched_conv2d() {
    let dev: TestDevice = Default::default();
    let x: Tensor<Rank3<3, 28, 28>, TestDtype, _> = dev.sample_normal();
    let w: Tensor<Rank4<5, 3, 6, 6>, TestDtype, _> = dev.sample_normal();

    let y: Tensor<Rank3<5, 9, 9>, _, _, _> =
        (x.leaky_trace(), w.clone()).conv2d(Const::<3>, Const::<2>, Const::<1>, Const::<1>);
    let y0 = y.retaped::<NoneTape>();
    let grads0 = y.square().mean().backward();
    let x0 = grads0.get(&x);
    let w0 = grads0.get(&w);

    let x = x
        .broadcast::<Rank4<10, 3, 28, 28>, _>()
        .reshape::<Rank4<10, 3, 28, 28>>();
    assert_eq!(x.strides, x.shape.strides());

    let y: Tensor<Rank4<10, 5, 9, 9>, _, _, _> =
        (x.leaky_trace(), w.clone()).conv2d(Const::<3>, Const::<2>, Const::<1>, Const::<1>);
    for i in 0..10 {
        assert_close_to_tensor!(y0, y.retaped::<NoneTape>().select(dev.tensor(i)));
    }

    let grads = y.square().mean().backward();

    assert_close_to_tensor!(w0, grads.get(&w), 1e-3);

    let x_grad = grads.get(&x) * 10.0;
    for i in 0..10 {
        assert_close_to_tensor!(x0, x_grad.clone().select(dev.tensor(i)));
    }
}
