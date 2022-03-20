use super::{structs::*, traits::*};
use crate::prelude::GradientTape;

fn unique_id() -> usize {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl TapeHolder for WithTape {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, mut update_fn: F) {
        update_fn(&mut self.0)
    }
}

impl TapeHolder for NoTape {
    fn update_with<F: FnMut(&mut Box<GradientTape>)>(&mut self, _update_fn: F) {}
}

macro_rules! tensor_impl {
    ($typename:ident, [$($const_names:tt),*]) => {
        impl<$(const $const_names: usize, )* HIn: TapeHolder, HOut: TapeHolder> CanReplaceTapeHolder<HOut> for $typename<$($const_names, )* HIn> {
            type Output = $typename<$($const_names, )* HOut>;

            fn replace_tape_holder(self, tape: HOut) -> Self::Output {
                Self::Output { id: self.id, data: self.data, tape }
            }
        }

        impl<$(const $const_names: usize),*> TensorCreator for $typename<$($const_names, )* NoTape> {
            fn new(data: ndarray::Array<f32, Self::Dimension>) -> Self {
                Self { id: unique_id(), data, tape: NoTape::default() }
            }
        }

        impl<$(const $const_names: usize),*> TapeCreator for $typename<$($const_names, )* NoTape> {
            fn with_tape(&self) -> Self::WithTape {
                Self::WithTape { id: self.id, data: self.data.clone(), tape: WithTape::default() }
            }
        }

        impl<$(const $const_names: usize, )* H: TapeHolder> Tensor for $typename<$($const_names, )* H> {
            type TapeHolder = H;
            type NoTape = $typename<$($const_names, )* NoTape>;
            type WithTape = $typename<$($const_names, )* WithTape>;

            fn split_tape_holder(self) -> (Self::NoTape, Self::TapeHolder) {
                (
                    Self::NoTape { id: self.id, data: self.data, tape: NoTape::default() },
                    self.tape,
                )
            }
        }
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [N]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

// TODO move this somewhere else
pub fn backward<T: Tensor<TapeHolder = WithTape>>(t: T) -> Box<GradientTape> {
    let id = t.id();
    let (_, mut tape_holder) = t.split_tape_holder();
    tape_holder.0.backward(id);
    tape_holder.0
}
