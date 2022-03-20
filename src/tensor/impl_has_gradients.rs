use super::{structs::*, traits::*};
use crate::prelude::GradientTape;
use std::ops::SubAssign;

fn update_with_tape<T: HasUniqueId + IsShapedArray>(t: &mut T, tape: &GradientTape) {
    let gradient = tape.gradient_for(t.id());
    t.mut_data().sub_assign(gradient);
}

impl<Tape> CanUpdateWithTape for Tensor0D<Tape> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}

impl<const N: usize, Tape> CanUpdateWithTape for Tensor1D<N, Tape> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}

impl<const M: usize, const N: usize, Tape> CanUpdateWithTape for Tensor2D<M, N, Tape> {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}

impl<const M: usize, const N: usize, const O: usize, Tape> CanUpdateWithTape
    for Tensor3D<M, N, O, Tape>
{
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, Tape> CanUpdateWithTape
    for Tensor4D<M, N, O, P, Tape>
{
    fn update_with_tape(&mut self, tape: &GradientTape) {
        update_with_tape(self, tape);
    }
}
