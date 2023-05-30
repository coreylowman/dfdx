use super::fixup_batchnorms::FixupBatchnorms;
use super::SqueezeAndExcite;
use super::E;
use dfdx::prelude::*;

pub enum Assert<const CHECK: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

pub type Block<
    const IN_CHAN: usize,
    const EXPAND_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL: usize,
    const STRIDE: usize,
    const USE_SE: bool,
    NonLinearity,
> = BlockInner<
    IN_CHAN,
    EXPAND_CHAN,
    OUT_CHAN,
    KERNEL,
    STRIDE,
    { (IN_CHAN == OUT_CHAN) & (STRIDE == 1) },
    USE_SE,
    { EXPAND_CHAN == OUT_CHAN },
    NonLinearity,
>;

pub struct BlockInner<
    const IN_CHAN: usize,
    const EXPAND_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL: usize,
    const STRIDE: usize,
    const USE_RESIDUAL: bool,
    const USE_SE: bool,
    const USE_SHORTCUT: bool,
    NonLinearity: BuildOnDevice<AutoDevice, E>,
>(std::marker::PhantomData<NonLinearity>);

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > BuildOnDevice<AutoDevice, E>
    for BlockInner<
        IN_CHAN,
        EXPAND_CHAN,
        OUT_CHAN,
        KERNEL,
        STRIDE,
        false,
        false,
        false,
        NonLinearity,
    >
where
    [(); (KERNEL - 1) / 2]:,
{
    type Built = <(
        (
            Conv2D<IN_CHAN, EXPAND_CHAN, 1>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        (
            Conv2D<1, EXPAND_CHAN, KERNEL, STRIDE, { (KERNEL - 1) / 2 }, 1, EXPAND_CHAN>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        (Conv2D<EXPAND_CHAN, OUT_CHAN, 1>, BatchNorm2D<OUT_CHAN>),
    ) as BuildOnDevice<AutoDevice, E>>::Built;
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > FixupBatchnorms<AutoDevice, E>
    for BlockInner<
        IN_CHAN,
        EXPAND_CHAN,
        OUT_CHAN,
        KERNEL,
        STRIDE,
        false,
        false,
        false,
        NonLinearity,
    >
where
    [(); (KERNEL - 1) / 2]:,
{
    fn fixup_batchnorms(m: &mut <Self as BuildOnDevice<AutoDevice, E>>::Built) {
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.0 .1);
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.1 .1);
        BatchNorm2D::<OUT_CHAN>::fixup_batchnorms(&mut m.2 .1);
    }
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > BuildOnDevice<AutoDevice, E>
    for BlockInner<IN_CHAN, EXPAND_CHAN, OUT_CHAN, KERNEL, STRIDE, false, true, false, NonLinearity>
where
    [(); (KERNEL - 1) / 2]:,
    [(); (((EXPAND_CHAN / 4) + 4) / 8) * 8]:,
{
    type Built = <(
        (
            Conv2D<IN_CHAN, EXPAND_CHAN, 1>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        (
            Conv2D<1, EXPAND_CHAN, KERNEL, STRIDE, { (KERNEL - 1) / 2 }, 1, EXPAND_CHAN>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        SqueezeAndExcite<EXPAND_CHAN, { (((EXPAND_CHAN / 4) + 4) / 8) * 8 }>,
        (Conv2D<EXPAND_CHAN, OUT_CHAN, 1>, BatchNorm2D<OUT_CHAN>),
    ) as BuildOnDevice<AutoDevice, E>>::Built;
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > FixupBatchnorms<AutoDevice, E>
    for BlockInner<IN_CHAN, EXPAND_CHAN, OUT_CHAN, KERNEL, STRIDE, false, true, false, NonLinearity>
where
    [(); (KERNEL - 1) / 2]:,
    [(); (((EXPAND_CHAN / 4) + 4) / 8) * 8]:,
{
    fn fixup_batchnorms(m: &mut <Self as BuildOnDevice<AutoDevice, E>>::Built) {
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.0 .1);
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.1 .1);
        BatchNorm2D::<OUT_CHAN>::fixup_batchnorms(&mut m.3 .1);
    }
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > BuildOnDevice<AutoDevice, E>
    for BlockInner<IN_CHAN, EXPAND_CHAN, OUT_CHAN, KERNEL, STRIDE, true, false, false, NonLinearity>
where
    [(); (KERNEL - 1) / 2]:,
{
    type Built = <Residual<(
        (
            Conv2D<IN_CHAN, EXPAND_CHAN, 1>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        (
            Conv2D<1, EXPAND_CHAN, KERNEL, STRIDE, { (KERNEL - 1) / 2 }, 1, EXPAND_CHAN>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        (Conv2D<EXPAND_CHAN, OUT_CHAN, 1>, BatchNorm2D<OUT_CHAN>),
    )> as BuildOnDevice<AutoDevice, E>>::Built;
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > FixupBatchnorms<AutoDevice, E>
    for BlockInner<IN_CHAN, EXPAND_CHAN, OUT_CHAN, KERNEL, STRIDE, true, false, false, NonLinearity>
where
    [(); (KERNEL - 1) / 2]:,
{
    fn fixup_batchnorms(m: &mut <Self as BuildOnDevice<AutoDevice, E>>::Built) {
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.0 .0 .1);
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.0 .1 .1);
        BatchNorm2D::<OUT_CHAN>::fixup_batchnorms(&mut m.0 .2 .1);
    }
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > BuildOnDevice<AutoDevice, E>
    for BlockInner<IN_CHAN, EXPAND_CHAN, OUT_CHAN, KERNEL, STRIDE, true, true, false, NonLinearity>
where
    [(); (KERNEL - 1) / 2]:,
    [(); (((EXPAND_CHAN / 4) + 4) / 8) * 8]:,
{
    type Built = <Residual<(
        (
            Conv2D<IN_CHAN, EXPAND_CHAN, 1>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        (
            Conv2D<1, EXPAND_CHAN, KERNEL, STRIDE, { (KERNEL - 1) / 2 }, 1, EXPAND_CHAN>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        SqueezeAndExcite<EXPAND_CHAN, { (((EXPAND_CHAN / 4) + 4) / 8) * 8 }>,
        (Conv2D<EXPAND_CHAN, OUT_CHAN, 1>, BatchNorm2D<OUT_CHAN>),
    )> as BuildOnDevice<AutoDevice, E>>::Built;
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > FixupBatchnorms<AutoDevice, E>
    for BlockInner<IN_CHAN, EXPAND_CHAN, OUT_CHAN, KERNEL, STRIDE, true, true, false, NonLinearity>
where
    [(); (KERNEL - 1) / 2]:,
    [(); (((EXPAND_CHAN / 4) + 4) / 8) * 8]:,
{
    fn fixup_batchnorms(m: &mut <Self as BuildOnDevice<AutoDevice, E>>::Built) {
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.0 .0 .1);
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.0 .1 .1);
        BatchNorm2D::<OUT_CHAN>::fixup_batchnorms(&mut m.0 .3 .1);
    }
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > BuildOnDevice<AutoDevice, E>
    for BlockInner<IN_CHAN, EXPAND_CHAN, OUT_CHAN, KERNEL, STRIDE, false, true, true, NonLinearity>
where
    [(); (KERNEL - 1) / 2]:,
    [(); (((EXPAND_CHAN / 4) + 4) / 8) * 8]:,
    Assert<{ IN_CHAN == EXPAND_CHAN }>: IsTrue,
{
    type Built = <(
        (
            Conv2D<1, EXPAND_CHAN, KERNEL, STRIDE, { (KERNEL - 1) / 2 }, 1, EXPAND_CHAN>,
            BatchNorm2D<EXPAND_CHAN>,
            NonLinearity,
        ),
        SqueezeAndExcite<EXPAND_CHAN, { (((EXPAND_CHAN / 4) + 4) / 8) * 8 }>,
        (Conv2D<EXPAND_CHAN, OUT_CHAN, 1>, BatchNorm2D<OUT_CHAN>),
    ) as BuildOnDevice<AutoDevice, E>>::Built;
}

impl<
        const IN_CHAN: usize,
        const EXPAND_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL: usize,
        const STRIDE: usize,
        NonLinearity: BuildOnDevice<AutoDevice, E>,
    > FixupBatchnorms<AutoDevice, E>
    for BlockInner<IN_CHAN, EXPAND_CHAN, OUT_CHAN, KERNEL, STRIDE, false, true, true, NonLinearity>
where
    [(); (KERNEL - 1) / 2]:,
    [(); (((EXPAND_CHAN / 4) + 4) / 8) * 8]:,
    Assert<{ IN_CHAN == EXPAND_CHAN }>: IsTrue,
{
    fn fixup_batchnorms(m: &mut <Self as BuildOnDevice<AutoDevice, E>>::Built) {
        BatchNorm2D::<EXPAND_CHAN>::fixup_batchnorms(&mut m.0 .1);
        BatchNorm2D::<OUT_CHAN>::fixup_batchnorms(&mut m.2 .1);
    }
}
