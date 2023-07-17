#![cfg_attr(
    all(feature = "test-integrations", feature = "nightly"),
    feature(generic_const_exprs)
)]

#[cfg(all(feature = "test-integrations", feature = "nightly"))]
#[path = "mobilenet_v3"]
mod model {
    use block::Block;
    use dfdx::prelude::*;
    use squeeze_and_excite::builder::SqueezeAndExcite;

    mod block;
    pub mod fixup_batchnorms;
    mod squeeze_and_excite;

    type E = f32;

    type Head = (Conv2D<3, 16, 3, 2, 1>, BatchNorm2D<16>, HardSwish);

    type Tail<const NUM_CLASSES: usize> = (
        (Conv2D<96, 576, 1>, BatchNorm2D<576>, HardSwish),
        AvgPoolGlobal,
        (Linear<576, 1024>, HardSwish),
        DropoutOneIn<5>,
        Linear<1024, NUM_CLASSES>,
    );

    pub type MobilenetV3Small<const NUM_CLASSES: usize> = (
        Head,
        (
            Block<16, 16, 16, 3, 2, true, ReLU>,
            Block<16, 72, 24, 3, 2, false, ReLU>,
            Block<24, 88, 24, 3, 1, false, ReLU>,
        ),
        (
            Block<24, 96, 40, 5, 2, true, HardSwish>,
            Block<40, 240, 40, 5, 1, true, HardSwish>,
            Block<40, 240, 40, 5, 1, true, HardSwish>,
        ),
        (
            Block<40, 120, 48, 5, 1, true, HardSwish>,
            Block<48, 144, 48, 5, 1, true, HardSwish>,
        ),
        (
            Block<48, 288, 96, 5, 2, true, HardSwish>,
            Block<96, 576, 96, 5, 1, true, HardSwish>,
            Block<96, 576, 96, 5, 1, true, HardSwish>,
        ),
        Tail<NUM_CLASSES>,
    );
}

#[test]
#[cfg(all(feature = "test-integrations", feature = "nightly"))]
fn test_mobilenet_v3_small_inference() {
    use dfdx::prelude::*;
    use model::fixup_batchnorms::FixupBatchnorms;

    let dev: AutoDevice = Default::default();
    let mut model = dev.build_module::<model::MobilenetV3Small<1000>, f32>();
    model.load("./tests/mobilenet_v3_small.npz").unwrap();

    // temp hack until #485 is fixed
    model::MobilenetV3Small::<1000>::fixup_batchnorms(&mut model);

    let mut x: Tensor<Rank4<10, 3, 224, 224>, f32, _> = dev.zeros();
    x.load_from_npy("./tests/mobilenet_v3_small_x.npy").unwrap();

    let mut y: Tensor<Rank2<10, 1000>, f32, _> = dev.zeros();
    y.load_from_npy("./tests/mobilenet_v3_small_y.npy").unwrap();

    let p = model.forward(x.clone());

    let p = p.array();
    let y = y.array();

    for i in 0..10 {
        for j in 0..1000 {
            assert!(
                (p[i][j] - y[i][j]).abs() <= 1e-4,
                "p[{i}][{j}]={} y[{i}][{j}]={}",
                p[i][j],
                y[i][j],
            );
        }
    }

    let p2 = model.forward(x).array();
    for i in 0..10 {
        for j in 0..1000 {
            assert!(
                (p[i][j] - p2[i][j]).abs() <= 1e-4,
                "p[{i}][{j}]={} p2[{i}][{j}]={}",
                p[i][j],
                p2[i][j],
            );
        }
    }
}
