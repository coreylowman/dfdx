use super::*;
use crate::{tensor_ops::*, tests::*};

#[test]
/// Produced by
/// ```python
/// q = torch.nn.Conv1d(1, 2, 2)
/// x = torch.sample_normal(1, 2, 3, requires_grad=True)
/// q(x).exp().mean().backward()
/// ```
fn test_conv1d_default_stride_and_padding() {
    let dev: TestDevice = Default::default();
    let weight = dev
        .tensor([[[0.02272552, 0.5804308]], [[0.7020492, 0.05399668]]])
        .to_dtype::<TestDtype>();
    let bias = dev.tensor([-0.3903233, 0.45088166]).to_dtype::<TestDtype>();
    let x = dev
        .tensor([[-1.6405383, 1.1242371]])
        .to_dtype::<TestDtype>();
    let result = (x.leaky_trace(), weight.clone())
        .conv1d(Const::<1>, Const::<0>, Const::<1>, Const::<1>)
        + bias.leaky_trace().broadcast::<_, _>();
    assert_close_to_literal!(result, [[0.2249364], [-0.6401519]]);
    let g = result.exp().mean().backward();
    assert_close_to_literal!(g.get(&x), [[0.19929343, 0.37765408]]);
    assert_close_to_literal!(
        g.get(&weight),
        [[[-1.0271764, 0.70390904]], [[-0.43245602, 0.2963558]]]
    );
    assert_close_to_literal!(g.get(&bias), [0.6261215, 0.26360616]);
}

#[test]
/// Produced by
/// ```python
/// q = torch.nn.Conv1d(1, 2, 2, stride=2)
/// x = torch.sample_normal(1, 2, 3, requires_grad=True)
/// q(x).exp().mean().backward()
/// ```
fn test_conv1d_stride_2() {
    let dev: TestDevice = Default::default();
    let weight = dev
        .tensor([[[-0.4296614, 0.27693725]], [[-0.3809104, 0.19169092]]])
        .to_dtype::<TestDtype>();
    let bias = dev
        .tensor([-0.29623124, -0.09120554])
        .to_dtype::<TestDtype>();
    let x = dev
        .tensor([[-0.31544453, 0.47184715]])
        .to_dtype::<TestDtype>();

    let result = (x.leaky_trace(), weight.clone())
        .conv1d(Const::<2>, Const::<0>, Const::<1>, Const::<1>)
        + bias.leaky_trace().broadcast::<_, _>();
    assert_close_to_literal!(result, [[-0.03002486], [0.11939937]]);

    let g = result.exp().mean().backward();

    assert_close_to_literal!(g.get(&x), [[-0.42308503, 0.2423735]]);

    assert_close_to_literal!(
        g.get(&weight),
        [[[-0.15305707, 0.22894529]], [[-0.17772458, 0.26584336]]]
    );

    assert_close_to_literal!(g.get(&bias), [0.48521072, 0.5634099]);
}

#[test]
fn test_conv1d_padding_1() {
    let dev: TestDevice = Default::default();
    let weight = dev
        .tensor([
            [[0.45220423, -0.3358205], [0.16038167, 0.09695387]],
            [[0.19551754, -0.3192072], [-0.49848652, -0.49257886]],
            [[0.21106702, 0.40513265], [0.08618081, -0.15866321]],
        ])
        .to_dtype::<TestDtype>();
    let bias = dev
        .tensor([-0.01069266, 0.22007078, -0.4849882])
        .to_dtype::<TestDtype>();
    let x = dev
        .tensor([[0.10943512, -1.7794625], [1.1263468, 0.5267281]])
        .to_dtype::<TestDtype>();

    let result = (x.leaky_trace(), weight.clone())
        .conv1d(Const::<1>, Const::<1>, Const::<1>, Const::<1>)
        + bias.leaky_trace().broadcast::<_, Axis<1>>();

    assert_close_to_literal!(
        result,
        [
            [0.06176047, 0.868088, -0.7308956],
            [-0.36967635, -0.01143935, -0.3904122],
            [-0.6193623, -1.1693113, -0.8151802]
        ]
    );

    let g = result.exp().mean().backward();

    assert_close_to_literal!(
        g.get(&x),
        [[0.10849564, -0.06070386], [-0.04517691, -0.05858667]]
    );

    assert_close_to_literal!(
        g.get(&weight),
        [
            [[-0.06622871, -0.45809978], [0.32632908, 0.27255058]],
            [[-0.12179004, -0.1870675], [0.16333483, 0.1443328]],
            [[-0.08372552, -0.05486213], [0.06477002, 0.08554335]]
        ]
    );

    assert_close_to_literal!(g.get(&bias), [0.43639296, 0.26181796, 0.14349198]);
}

#[test]
fn test_conv1d_stride_3_padding_4() {
    let dev: TestDevice = Default::default();
    let weight = dev
        .tensor([
            [[-0.4961109, -0.41855216, -0.31035745]],
            [[-0.28658125, 0.09752917, -0.4264508]],
        ])
        .to_dtype::<TestDtype>();
    let bias = dev.tensor([0.04796273, 0.17420131]).to_dtype::<TestDtype>();
    let x = dev
        .tensor([[0.09930344, 1.0408987]])
        .to_dtype::<TestDtype>();

    let result = (x.leaky_trace(), weight.clone())
        .conv1d(Const::<3>, Const::<4>, Const::<1>, Const::<1>)
        + bias.leaky_trace().broadcast::<_, _>();

    assert_close_to_literal!(
        result,
        [
            [0.04796273, -0.31665158, 0.04796273],
            [0.17420131, -0.26000577, 0.17420131]
        ]
    );

    let g = result.exp().mean().backward();

    assert_close_to_literal!(g.get(&x), [[-0.03829185, -0.09248922]]);

    assert_close_to_literal!(
        g.get(&weight),
        [
            [[0., 0.01205849, 0.12639713]],
            [[0., 0.01276127, 0.13376366]]
        ]
    );

    assert_close_to_literal!(g.get(&bias), [0.4711413, 0.5252729]);
}

#[test]
fn test_conv1d_s4p3k2() {
    let dev = TestDevice::seed_from_u64(432);

    let weight: Tensor<Rank3<3, 5, 2>, TestDtype, _> = dev.sample_normal();
    println!("weight data {:?}", weight.as_vec());
    let bias: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
    println!("bias data {:?}", bias.as_vec());
    let x: Tensor<Rank2<5, 7>, TestDtype, _> = dev.sample_normal();
    println!("x data {:?}", x.as_vec());

    let out =
        (x.leaky_trace(), weight.clone()).conv1d(Const::<4>, Const::<3>, Const::<1>, Const::<1>);
    let out = out + bias.broadcast::<_, Axis<1>>();
    println!("out data {:?}, {:?}", out.as_vec(), out.shape());

    assert_close_to_literal!(
        out,
        [
            [0.44691145, 1.3863211, -2.0541177],
            [0.1279889, -0.96598804, 1.6030374],
            [-0.66274095, -1.2659106, -0.38152635],
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

#[test]
fn test_conv1d_dilated() {
    let dev: TestDevice = Default::default();
    let x = dev.tensor([[0., 1., 2., 4., 5.]]).to_dtype::<TestDtype>();
    let w = dev.tensor([[[0.1, 0.5]]]).to_dtype::<TestDtype>();
    let y = (x.leaky_trace(), w.clone()).conv1d(Const::<1>, Const::<0>, Const::<2>, Const::<1>);
    assert_close_to_literal!(y, [[1.0, 2.1, 2.7]]);
    let grads = y.mean().backward();
    assert_close_to_literal!(
        grads.get(&x),
        [[0.03333335, 0.03333335, 0.2, 0.1666667, 0.1666667],]
    );
    assert_close_to_literal!(grads.get(&w), [[[1.0, 11.0 / 3.0]]]);
}

#[test]
fn test_conv1d_grouped_forward() {
    const NUM_GROUPS: usize = 3;
    let dev: TestDevice = Default::default();
    let x: Tensor<Rank3<2, 9, 14>, TestDtype, _> = dev.sample_normal();
    let w: Tensor<Rank3<15, 3, 3>, TestDtype, _> = dev.sample_normal();

    let y = (x.leaky_trace(), w.clone()).conv1d(
        Const::<1>,
        Const::<0>,
        Const::<1>,
        Const::<NUM_GROUPS>,
    );

    for i in 0..NUM_GROUPS {
        let x_group = x
            .clone()
            .slice((.., 3 * i..3 * (i + 1), ..))
            .realize::<(Const<2>, Const<3>, Const<14>)>();
        let w_group = w
            .clone()
            .slice((5 * i..5 * (i + 1), .., ..))
            .realize::<(Const<5>, Const<3>, Const<3>)>();
        let y_group = (x_group, w_group).conv1d(Const::<1>, Const::<0>, Const::<1>, Const::<1>);
        let y_group_true = y
            .retaped::<NoneTape>()
            .slice((.., 5 * i..5 * (i + 1), ..))
            .realize::<(Const<2>, Const<5>, Const<12>)>();
        assert_close_to_tensor!(y_group, y_group_true);
    }

    let grads = y.exp().sum().backward();
    let x_grad = grads.get(&x);
    let w_grad = grads.get(&w);

    for i in 0..NUM_GROUPS {
        let x_group = x
            .clone()
            .slice((.., 3 * i..3 * (i + 1), ..))
            .realize::<(Const<2>, Const<3>, Const<14>)>();
        let w_group = w
            .clone()
            .slice((5 * i..5 * (i + 1), .., ..))
            .realize::<(Const<5>, Const<3>, Const<3>)>();
        let y_group = (x_group.leaky_trace(), w_group.clone())
            .conv1d(Const::<1>, Const::<0>, Const::<1>, Const::<1>);
        let grads = y_group.exp().sum().backward();

        let x_grad_group_true = x_grad
            .clone()
            .slice((.., 3 * i..3 * (i + 1), ..))
            .realize::<(Const<2>, Const<3>, Const<14>)>();
        let w_grad_group_true = w_grad
            .clone()
            .slice((5 * i..5 * (i + 1), .., ..))
            .realize::<(Const<5>, Const<3>, Const<3>)>();

        assert_close_to_tensor!(grads.get(&x_group), x_grad_group_true);
        assert_close_to_tensor!(grads.get(&w_group), w_grad_group_true);
    }
}
