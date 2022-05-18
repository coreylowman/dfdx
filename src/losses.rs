use crate::prelude::*;

pub fn mse_loss<T: Tensor>(pred: T, targ: &T::NoTape) -> Tensor0D<T::TapeHolder> {
    sub(targ, pred).square().mean()
}

pub fn mae_loss<T: Tensor>(pred: T, targ: &T::NoTape) -> Tensor0D<T::TapeHolder> {
    sub(targ, pred).abs().mean()
}

pub fn cross_entropy_with_logits_loss<T: Tensor + HasSoftmaxMethod>(
    logits: T,
    targ: &T::NoTape,
) -> Tensor0D<T::TapeHolder> {
    -mul(targ, logits.log_softmax()).mean()
}
