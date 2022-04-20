use super::add_unary_op;
use crate::prelude::*;
use ndarray::{prelude::*, Data, OwnedRepr};

fn softmax<S: Data<Elem = f32>, D: Dimension>(a: &ArrayBase<S, D>) -> ArrayBase<OwnedRepr<f32>, D> {
    let max = a.iter().cloned().reduce(f32::max).unwrap();
    let exp_data = a.mapv(|v| (v - max).exp());
    let total = exp_data.sum();
    exp_data.mapv(|v| v / total)
}

pub trait HasLogSoftmaxMethod {
    fn log_softmax(self) -> Self;
}

impl<const N: usize, H: TapeHolder> HasLogSoftmaxMethod for Tensor1D<N, H> {
    fn log_softmax(self) -> Self {
        let lse = self.data().mapv(f32::exp).sum().ln();
        let result = <Self as Tensor>::NoTape::new(self.data().mapv(|v| v - lse));
        let (t, mut tape_holder) = self.split_tape_holder();
        tape_holder.update_with(|tape| {
            let deriv = softmax(t.data());
            add_unary_op(tape, (&t, &result), deriv)
        });
        result.with_tape_holder(tape_holder)
    }
}

impl<const M: usize, const N: usize, H: TapeHolder> HasLogSoftmaxMethod for Tensor2D<M, N, H> {
    fn log_softmax(self) -> Self {
        let mut data = self.data().clone();
        for a in data.axis_iter_mut(Axis(0)) {
            let lse = a.mapv(f32::exp).sum().ln();
            let sa = a.mapv(|v| v - lse);
            ndarray::Zip::from(a).and(&sa).for_each(|q, &z| {
                *q = z;
            });
        }
        let result = <Self as Tensor>::NoTape::new(data);
        let (t, mut tape_holder) = self.split_tape_holder();
        tape_holder.update_with(|tape| {
            let mut deriv = t.data().clone();
            for a in deriv.axis_iter_mut(Axis(0)) {
                let sa = softmax(&a);
                ndarray::Zip::from(a).and(&sa).for_each(|q, &z| {
                    *q = z;
                });
            }
            add_unary_op(tape, (&t, &result), deriv)
        });
        result.with_tape_holder(tape_holder)
    }
}

pub trait HasSoftmaxMethod {
    fn softmax(self) -> Self;
}

impl<T> HasSoftmaxMethod for T
where
    T: HasLogSoftmaxMethod + HasExpMethod,
{
    fn softmax(self) -> Self {
        self.log_softmax().exp()
    }
}
