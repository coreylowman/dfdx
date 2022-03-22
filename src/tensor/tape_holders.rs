pub use super::*;
use crate::gradients::GradientTape;

#[derive(Default, Debug)]
pub struct WithTape(pub(crate) Box<GradientTape>);

#[derive(Default, Debug)]
pub struct NoTape;

pub trait TapeHolder {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, update_fn: F);
}

impl TapeHolder for WithTape {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, mut update_fn: F) {
        update_fn(&mut self.0)
    }
}

impl TapeHolder for NoTape {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, _update_fn: F) {}
}

pub trait HasTapeHolder<H: TapeHolder> {
    type Output;
    fn with_tape_holder(self, tape_holder: H) -> Self::Output;
}

impl<HIn: TapeHolder, HOut: TapeHolder> HasTapeHolder<HOut> for Tensor0D<HIn> {
    type Output = Tensor0D<HOut>;
    fn with_tape_holder(self, tape: HOut) -> Self::Output {
        Self::Output {
            id: self.id,
            data: self.data,
            tape,
        }
    }
}

impl<const N: usize, HIn: TapeHolder, HOut: TapeHolder> HasTapeHolder<HOut> for Tensor1D<N, HIn> {
    type Output = Tensor1D<N, HOut>;
    fn with_tape_holder(self, tape: HOut) -> Self::Output {
        Self::Output {
            id: self.id,
            data: self.data,
            tape,
        }
    }
}

impl<const M: usize, const N: usize, HIn: TapeHolder, HOut: TapeHolder> HasTapeHolder<HOut>
    for Tensor2D<M, N, HIn>
{
    type Output = Tensor2D<M, N, HOut>;
    fn with_tape_holder(self, tape: HOut) -> Self::Output {
        Self::Output {
            id: self.id,
            data: self.data,
            tape,
        }
    }
}

impl<const M: usize, const N: usize, const O: usize, HIn: TapeHolder, HOut: TapeHolder>
    HasTapeHolder<HOut> for Tensor3D<M, N, O, HIn>
{
    type Output = Tensor3D<M, N, O, HOut>;
    fn with_tape_holder(self, tape: HOut) -> Self::Output {
        Self::Output {
            id: self.id,
            data: self.data,
            tape,
        }
    }
}
impl<
        const M: usize,
        const N: usize,
        const O: usize,
        const P: usize,
        HIn: TapeHolder,
        HOut: TapeHolder,
    > HasTapeHolder<HOut> for Tensor4D<M, N, O, P, HIn>
{
    type Output = Tensor4D<M, N, O, P, HOut>;
    fn with_tape_holder(self, tape: HOut) -> Self::Output {
        Self::Output {
            id: self.id,
            data: self.data,
            tape,
        }
    }
}
