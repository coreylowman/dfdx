use super::*;
use crate::gradients::{NoneTape, OwnedTape};

/// Transforms a [NoneTape] tensor to an [OwnedTape] tensor by cloning.
/// Clones `t` using, and then inserts [OwnedTape] as the tape.
///
/// See [traced()] for version that takes ownership of `t`.
pub fn trace<T: Tensor<Tape = OwnedTape>>(t: &T::NoTape) -> T {
    traced(t.clone())
}

/// Transforms a [NoneTape] tensor to an [OwnedTape] by directly inserting a
/// new [OwnedTape] into `t`.
///
/// See [trace()] for version that copies `t`.
pub fn traced<T: Tensor<Tape = OwnedTape>>(t: T::NoTape) -> T {
    t.put_tape(OwnedTape::default())
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )*> $typename<$($Vs, )* NoneTape> {
    /// Clones `self` and returns a copy with [OwnedTape] as the [crate::gradients::Tape].
    ///
    /// See `traced` for a version that takes ownership of the tensor.
    pub fn trace(&self) -> $typename<$($Vs, )* OwnedTape> {
        trace(self)
    }

    /// Takes ownership of `self` and inserts [OwnedTape] as the [crate::gradients::Tape].
    pub fn traced(self) -> $typename<$($Vs, )* OwnedTape> {
        traced(self)
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
    fn test_trace() {
        let t1: Tensor1D<32> = TensorCreator::zeros();
        let t2: Tensor1D<32, OwnedTape> = trace(&t1);
        assert_eq!(t1.id, t2.id);
    }

    #[test]
    fn test_traced() {
        let t1: Tensor1D<32> = TensorCreator::zeros();
        let t1_id = t1.id;
        let t2: Tensor1D<32, OwnedTape> = traced(t1);
        assert_eq!(t1_id, t2.id);
    }

    #[test]
    fn test_trace_split() {
        let t1: Tensor1D<32> = TensorCreator::zeros();
        let t2: Tensor1D<32, OwnedTape> = t1.trace();
        let (t3, tape): (Tensor1D<32, NoneTape>, OwnedTape) = t2.split_tape();
        let _: Tensor1D<32, OwnedTape> = t3.put_tape(tape);
    }
}
