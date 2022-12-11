#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "nightly")]
fn main() {
    use dfdx::prelude::*;
    use std::time::Instant;

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

    type Resnet18<const NUM_CLASSES: usize> = (
        Head,
        (BasicBlock<64>, ReLU, BasicBlock<64>, ReLU),
        (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
        (Downsample<128, 256>, ReLU, BasicBlock<256>, ReLU),
        (Downsample<256, 512>, ReLU, BasicBlock<512>, ReLU),
        (AvgPoolGlobal, Linear<512, NUM_CLASSES>),
    );

    let dev: Cpu = Default::default();
    let x = dev.randn::<Rank3<3, 224, 224>>();
    let m: Resnet18<1000> = dev.build();
    for _ in 0.. {
        let start = Instant::now();
        let _ = m.forward(x.clone());
        println!("{:?}", start.elapsed());
    }
}

#[cfg(not(feature = "nightly"))]
fn main() {
    panic!("Run with `cargo +nightly run ...` to run this example.");
}
