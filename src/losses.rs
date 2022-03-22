use crate::prelude::*;

pub fn mse_loss<T: Tensor>(pred: T, targ: &T::NoTape) -> Tensor0D<T::TapeHolder> {
    sub(targ, pred).square().mean()
}

pub fn mae_loss<T: Tensor>(pred: T, targ: &T::NoTape) -> Tensor0D<T::TapeHolder> {
    sub(targ, pred).abs().mean()
}
