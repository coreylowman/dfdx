use super::*;
use crate::prelude::GradientTape;
use std::ops::SubAssign;

pub trait CanUpdateWithTape {
    fn update_with_tape(&mut self, tape: &GradientTape);
}

fn update_with_tape<T: HasUniqueId + IsShapedArray>(t: &mut T, tape: &GradientTape) {
    let gradient = tape.gradient_for(t.id());
    t.mut_data().sub_assign(gradient);
}

impl<H> CanUpdateWithTape for Tensor0D<H> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}

impl<const N: usize, H> CanUpdateWithTape for Tensor1D<N, H> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}

impl<const M: usize, const N: usize, H> CanUpdateWithTape for Tensor2D<M, N, H> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}

impl<const M: usize, const N: usize, const O: usize, H> CanUpdateWithTape for Tensor3D<M, N, O, H> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H> CanUpdateWithTape
    for Tensor4D<M, N, O, P, H>
{
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}
