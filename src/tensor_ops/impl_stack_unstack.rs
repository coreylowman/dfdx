use super::utils::move_tape_and_add_backward_op;
use crate::arrays::{AllAxes, Axes2, Axes3, Axis, HasArrayType};
use crate::devices::{AddAccum, CopyAccum, Cpu, DeviceReduce};
use crate::gradients::Tape;
use crate::prelude::*;
use std::vec::Vec;

/// Unstack self into a list of `T` along `Axes`. Opposite of [Stack].
pub trait UnstackTo<T, Axes> {
    /// Broadcast `self` into `T`. This can be used to broadcast 1, 2, 3, and 4 axes.
    ///
    /// Examples:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// // broadcast axis 1
    /// let _: Tensor3D<3, 5, 7> = Tensor2D::<3, 7>::zeros().broadcast();
    ///
    /// // broadcast axes 0, 1
    /// let _: Tensor3D<7, 5, 3> = Tensor1D::<3>::zeros().broadcast();
    ///
    /// // broadcast axes 1, 2, 3
    /// let _: Tensor4D<3, 5, 7, 9> = Tensor1D::<3>::zeros().broadcast();
    /// ```
    fn broadcast(self) -> T;
}

/// Stack a list of Tensors along Axes to return a tensor with one more dimension. Opposite of [UnstackTo].
///
/// This trait can't be used directly as it doesn't contain any methods. Instead
/// it is used by methods to specify the input type must be able to have it's axes
/// reduced.
pub trait Stack<L, T, Axes>
    where
        L: AsRef<[T]>,
        T: Sized + Tensor<Dtype = f32> {

    /// The resulting tensor type.
    /// This can be broadcast into Self via [BroadcastTo].
    type Stacked: UnstackTo<T, Axes> + Tensor<Tape = T::Tape, Dtype = T::Dtype>;
    // type DeviceR: DeviceReduce<T::Array, Axes, Reduced = <T::Reduced as HasArrayType>::Array>;

    fn stack(self: L) -> Self::Stacked;
}

/// Stack `Axes` of `Self` to produce a `T`
// pub trait StackTo<L, T, Axes>: Stack<Axes, Stacked = T> {}




impl<const M: usize, const N: usize, H: Tape> Stack<_, Tensor1D<M, H>, Axis<1>> for dyn AsRef<[Tensor1D<M, H>]> {
    type Stacked = Tensor2D<M, N, H>;

    fn stack(self: [Tensor1D<M, H>; N]) -> Tensor2D<M, N, H> {
        let mut result = <Tensor2D<M, N, H> as Tensor>::NoTape::zeros();
        for n in 0..N {
            let mut result_slice: Tensor1D<M, H> = result.select(&[n]);
            <Cpu as DeviceReduce<_, Axis<1>>>::broadcast_into_no_reset::<CopyAccum>(result_slice.mut_data(), self[n].data());
        }
        result
    }

    // fn stack(tensors: [Tensor1D<M, H>; N]) -> Tensor2D<M, N, H> {
    //     let mut result = <Tensor2D<M, N, H> as Tensor>::NoTape::zeros();
    //     for n in 0..N {
    //         let mut result_slice: Tensor1D<M, H> = result.select(&[n]);
    //         <Cpu as DeviceReduce<_, Axis<1>>>::broadcast_into_no_reset::<CopyAccum>(result_slice.mut_data(), tensors[n].data());
    //     }
    //     result
    // }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::AssertClose;
    use rand::thread_rng;
    use std::dbg;

    #[test]
    fn test_valid_1d_broadcasts() {
        let mut tensors: [Tensor1D<3>; 2] = [
            Tensor1D::zeros(),
            Tensor1D::zeros()
        ];
        let stacked = Stack::stack(tensors);
        dbg!(stacked);
    }
}
