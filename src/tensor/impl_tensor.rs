use crate::arrays::HasArrayType;
use crate::gradients::{CanUpdateWithGradients, NoneTape, Tape};
use crate::prelude::*;
use crate::unique_id::{internal, unique_id, HasUniqueId};

/// The main tensor trait. A tensor consists of mainly 1. an array, 2. a device, 3. a unique id.
pub trait Tensor:
    HasArrayType
    + HasArrayData
    + HasDevice
    + CanUpdateWithGradients
    + HasUniqueId
    + IntoPhantom
    + internal::ResetId
{
    /// The [Tape] this tensor owns.
    type Tape: Tape;

    /// This tensor but with [NoneTape].
    type NoTape: 'static
        + Tensor<Array = Self::Array, Dtype = Self::Dtype, Tape = NoneTape, NoTape = Self::NoTape>
        // NOTE: we only want to be able to create NoneTape tensors
        + TensorCreator
        // NOTE: Adding this restriction means we can put the tape from Self into the Self::NoTape
        + PutTape<Self::Tape, Output = Self>
        + Clone;

    /// Removes whatever Tape this tensor has and returns itself without a tape.
    fn split_tape(self) -> (Self::NoTape, Self::Tape);

    /// Clones self and initializes a new empty tape.
    fn with_new_tape(&self) -> Self;
}

macro_rules! tensor_impl {
    ($struct:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> internal::ResetId for $struct<$($Vs, )* H> {
    fn reset_id(&mut self) {
        self.id = unique_id();
    }
}
impl<$(const $Vs: usize, )* H: Tape> Tensor for $struct<$($Vs, )* H> {
    type Tape = H;
    type NoTape = $struct<$($Vs, )* NoneTape>;

    fn split_tape(self) -> (Self::NoTape, Self::Tape) {
        (
            Self::NoTape { id: self.id, data: self.data, tape: Default::default() },
            self.tape,
        )
    }

    fn with_new_tape(&self) -> Self {
        Self { id: self.id, data: self.data.clone(), tape: H::default() }
    }
}

impl<$(const $Vs: usize, )* H: Clone> Clone for $struct<$($Vs, )* H> {
    /// Clones the underlying id, data, and tape
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            data: self.data.clone(),
            tape: self.tape.clone(),
        }
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ids_with_clone() {
        let t1: Tensor1D<32> = TensorCreator::zeros();
        let t2: Tensor1D<32, NoneTape> = t1.clone();
        assert_eq!(t1.id, t2.id);
    }

    #[test]
    fn test_ids_with_split_and_put() {
        let t1: Tensor1D<32> = TensorCreator::zeros();
        let t1_id = t1.id;
        let (t2, tape) = t1.split_tape();
        assert_eq!(t2.id, t1_id);
        let t3 = t2.put_tape(tape);
        assert_eq!(t3.id, t1_id);
    }
}
