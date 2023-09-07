use super::*;
use crate::{tensor_ops::*, tests::*};

#[test]
fn test_conv1d_default_stride_and_padding() {
    let dev: TestDevice = Default::default();
    let weight = dev
        .tensor([[[0.46187001, 0.34853035]], [[-0.65229625, 0.04932868]]])
        .to_dtype::<TestDtype>();
    let x = dev
        .tensor([[1.33830595, 0.79225832]])
        .to_dtype::<TestDtype>();
    let result =
        (x.leaky_trace(), weight.clone()).conv1d(Const::<1>, Const::<0>, Const::<1>, Const::<1>);
    assert_close_to_literal!(result, [[0.89424950], [-0.83389091]]);
    let g = result.exp().mean().backward();
    assert_close_to_literal!(g.get(&x), [[0.42308712, 0.43687853]]);
    assert_close_to_literal!(
        g.get(&weight),
        [[[1.63641334, 0.96873373]], [[0.29065058, 0.17206106]]]
    );
}

#[test]
fn test_conv1d_all() {
    let dev: TestDevice = Default::default();
    let weight = dev
        .tensor([
            [[0.42317861, -0.41726643], [-0.39504033, 0.14706957]],
            [[0.04628271, -0.47788471], [0.46128953, 0.44733173]],
            [[-0.02302122, 0.28430778], [0.35562867, 0.14598244]],
        ])
        .to_dtype::<TestDtype>();
    let x = dev
        .tensor([
            [
                0.60798085,
                -0.04124952,
                -0.83411056,
                -0.35019234,
                0.59375221,
            ],
            [-0.53912121, 3.06222630, 0.98324996, 2.30645585, 1.33080256],
        ])
        .to_dtype::<TestDtype>();

    let result =
        (x.leaky_trace(), weight.clone()).conv1d(Const::<2>, Const::<1>, Const::<2>, Const::<1>);
    assert_close_to_literal!(
        result,
        [
            [0.46757236, -0.74182582, -1.05933702],
            [1.38954353, 2.60976601, 1.04773605],
            [0.43530372, 1.32710481, 0.82830369]
        ]
    );

    let g = result.exp().mean().backward();

    assert_close_to_literal!(
        g.get(&x),
        [
            [0.00000000, -0.15559882, 0.00000000, -0.59979343, 0.00000000],
            [0.00000000, 1.07552814, 0.00000000, 0.96608126, 0.00000000]
        ]
    );

    assert_close_to_literal!(
        g.get(&weight),
        [
            [[-0.01567238, -0.02584620], [0.25088674, 0.66512215]],
            [[-0.17325418, -0.54741162], [5.35664129, 4.84967136]],
            [[-0.10636187, -0.15377919], [1.86949003, 1.49200678]]
        ]
    );
}

#[test]
fn test_conv1d_all_grouped() {
    let dev: TestDevice = Default::default();
    let x = dev
        .tensor([
            [1.27495587, 0.09800160, -1.10806572, 0.71087748, 1.48600435],
            [
                -0.62874401,
                0.71909815,
                1.67691362,
                -1.93346322,
                -1.22098184,
            ],
        ])
        .to_dtype::<TestDtype>();
    let w = dev
        .tensor([
            [[-0.18984121, 0.45376354]],
            [[-0.53830475, 0.33164448]],
            [[-0.14025646, 0.52515274]],
            [[-0.61143655, -0.37861061]],
        ])
        .to_dtype::<TestDtype>();

    let y = (x.leaky_trace(), w.clone()).conv1d(Const::<2>, Const::<1>, Const::<2>, Const::<2>);
    assert_close_to_literal!(
        y,
        [
            [0.04446955, 0.30396554, -0.13495384],
            [0.03250169, 0.18300386, -0.38266873],
            [0.37763637, -1.11622167, 0.27118072],
            [-0.27225819, 0.29234678, 1.18219006]
        ]
    );

    let grads = y.exp().mean().backward();

    assert_close_to_literal!(
        grads.get(&x),
        [
            [0.00000000, -0.00722379, 0.00000000, 0.04001465, 0.00000000],
            [0.00000000, -0.03227153, 0.00000000, -0.20944442, 0.00000000]
        ]
    );
    assert_close_to_literal!(
        grads.get(&w),
        [
            [[0.06282897, 0.08882126]],
            [[0.05021068, 0.07957286]],
            [[-0.19168709, 0.03465047]],
            [[-0.44522732, -0.17019148]]
        ]
    );
}

#[test]
fn test_batched_conv1d() {
    let dev: TestDevice = Default::default();
    let x: Tensor<Rank2<3, 28>, TestDtype, _> = dev.sample_normal();
    let w: Tensor<Rank3<5, 3, 6>, TestDtype, _> = dev.sample_normal();

    let y: Tensor<Rank2<5, 9>, _, _, _> =
        (x.leaky_trace(), w.clone()).conv1d(Const::<3>, Const::<2>, Const::<1>, Const::<1>);
    let y0 = y.retaped::<NoneTape>();
    let grads0 = y.square().mean().backward();
    let x0 = grads0.get(&x);
    let w0 = grads0.get(&w);

    let x = x
        .broadcast::<Rank3<10, 3, 28>, _>()
        .reshape::<Rank3<10, 3, 28>>();
    assert_eq!(x.strides, x.shape.strides());

    let y: Tensor<Rank3<10, 5, 9>, _, _, _> =
        (x.leaky_trace(), w.clone()).conv1d(Const::<3>, Const::<2>, Const::<1>, Const::<1>);
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
