use super::Block;
use super::E;
use dfdx::prelude::*;

pub trait FixupBatchnorms<AutoDevice: Device<E>, E: Dtype>: BuildOnDevice<AutoDevice, E> {
    fn fixup_batchnorms(m: &mut <Self as BuildOnDevice<AutoDevice, E>>::Built);
}

impl<const C: usize> FixupBatchnorms<AutoDevice, E> for BatchNorm2D<C> {
    fn fixup_batchnorms(m: &mut <Self as BuildOnDevice<AutoDevice, E>>::Built) {
        m.epsilon = 0.001;
        m.momentum = 0.01;
    }
}

impl<const NUM_CLASSES: usize> FixupBatchnorms<AutoDevice, E>
    for super::MobilenetV3Small<NUM_CLASSES>
{
    fn fixup_batchnorms(m: &mut <Self as BuildOnDevice<AutoDevice, E>>::Built) {
        BatchNorm2D::fixup_batchnorms(&mut m.0 .1);
        BatchNorm2D::fixup_batchnorms(&mut m.1 .0 .0 .1);
        BatchNorm2D::fixup_batchnorms(&mut m.1 .0 .2 .1);

        Block::<16, 16, 16, 3, 2, true, ReLU>::fixup_batchnorms(&mut m.1 .0);
        Block::<16, 72, 24, 3, 2, false, ReLU>::fixup_batchnorms(&mut m.1 .1);
        Block::<24, 88, 24, 3, 1, false, ReLU>::fixup_batchnorms(&mut m.1 .2);
        Block::<24, 96, 40, 5, 2, true, HardSwish>::fixup_batchnorms(&mut m.2 .0);
        Block::<40, 240, 40, 5, 1, true, HardSwish>::fixup_batchnorms(&mut m.2 .1);
        Block::<40, 240, 40, 5, 1, true, HardSwish>::fixup_batchnorms(&mut m.2 .2);
        Block::<40, 120, 48, 5, 1, true, HardSwish>::fixup_batchnorms(&mut m.3 .0);
        Block::<48, 144, 48, 5, 1, true, HardSwish>::fixup_batchnorms(&mut m.3 .1);
        Block::<48, 288, 96, 5, 2, true, HardSwish>::fixup_batchnorms(&mut m.4 .0);
        Block::<96, 576, 96, 5, 1, true, HardSwish>::fixup_batchnorms(&mut m.4 .1);
        Block::<96, 576, 96, 5, 1, true, HardSwish>::fixup_batchnorms(&mut m.4 .2);

        BatchNorm2D::fixup_batchnorms(&mut m.5 .0 .1);
    }
}
