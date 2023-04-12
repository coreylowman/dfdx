#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(all(feature = "test-integrations", feature = "nightly"))]
mod model {
    use dfdx::prelude::*;

    type BasicBlock<const C: usize> = Residual<(
        Conv2D<C, C, 3, 1, 1>,
        BatchNorm2D<C>,
        ReLU,
        Conv2D<C, C, 3, 1, 1>,
        BatchNorm2D<C>,
    )>;

    type Downsample<const C: usize, const D: usize> = GeneralizedResidual<
        (
            Conv2D<C, D, 3, 2, 1>,
            BatchNorm2D<D>,
            ReLU,
            Conv2D<D, D, 3, 1, 1>,
            BatchNorm2D<D>,
        ),
        (Conv2D<C, D, 1, 2, 0>, BatchNorm2D<D>),
    >;

    type Head = (
        Conv2D<3, 64, 7, 2, 3>,
        BatchNorm2D<64>,
        ReLU,
        MaxPool2D<3, 2, 1>,
    );

    pub type Resnet18<const NUM_CLASSES: usize> = (
        Head,
        (BasicBlock<64>, ReLU, BasicBlock<64>, ReLU),
        (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
        (Downsample<128, 256>, ReLU, BasicBlock<256>, ReLU),
        (Downsample<256, 512>, ReLU, BasicBlock<512>, ReLU),
        (AvgPoolGlobal, Linear<512, NUM_CLASSES>),
    );
}

#[test]
#[cfg(all(feature = "test-integrations", feature = "nightly"))]
fn test_resnet18_f32_inference() {
    use dfdx::prelude::*;
    let dev: AutoDevice = Default::default();
    let mut model = dev.build_module::<model::Resnet18<1000>, f32>();
    model.load("./tests/resnet18.npz").unwrap();

    let mut x: Tensor<Rank4<10, 3, 224, 224>, f32, _> = dev.zeros();
    x.load_from_npy("./tests/resnet18_x.npy").unwrap();

    let mut y: Tensor<Rank2<10, 1000>, f32, _> = dev.zeros();
    y.load_from_npy("./tests/resnet18_y.npy").unwrap();

    let p = model.forward(x.clone());

    let p = p.array();
    let y = y.array();

    for i in 0..10 {
        for j in 0..1000 {
            assert!(
                (p[i][j] - y[i][j]).abs() <= 1e-5,
                "p[{i}][{j}]={} y[{i}][{j}]={}",
                p[i][j],
                y[i][j]
            );
        }
    }

    let p2 = model.forward(x);
    assert_eq!(p, p2.array());
}
