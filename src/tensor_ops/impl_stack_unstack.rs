use super::utils::move_tape_and_add_backward_op;
use crate::arrays::{Axis};
use crate::devices::{AddAccum, CopyAccum, Cpu, DeviceReduce, FillElements, SelectAx1};
use crate::tensor_ops::SelectTo;
use crate::gradients::Tape;
use crate::prelude::*;
use crate::prelude::utils::move_tape_and_add_backward_listop;

/// Unstack self into a list of `T` along `Axes`. Opposite of [Stack].
pub trait UnstackTo<L, T, Axes>
    where L: AsRef<[T]>
    {
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
    fn unstack(self) -> L;
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
    type Stacked: UnstackTo<L, T, Axes> + Tensor<Tape = T::Tape, Dtype = T::Dtype>;
    // type DeviceR: DeviceReduce<T::Array, Axes, Reduced = <T::Reduced as HasArrayType>::Array>;

    fn stack(self) -> Self::Stacked;
}

/// Stack `Axes` of `Self` to produce a `T`
// pub trait StackTo<L, T, Axes>: Stack<Axes, Stacked = T> {}


impl<const M: usize, const N: usize, H: Tape> Stack<[Tensor1D<M, H>; N], Tensor1D<M, H>, Axis<1>> for [Tensor1D<M, H>; N] {
    type Stacked = Tensor2D<M, N, H>;

    fn stack(mut self) -> Tensor2D<M, N, H> {
        let mut result = <Tensor2D<M, N, H> as Tensor>::NoTape::zeros();
        for n in 0..N {
            // <T as HasDevice>::Device::select_axis(t.data(), indices, result.mut_data());
            let select_indices: [usize; M] = [n; M];
            let mut result_slice: Tensor1D<M> = SelectTo::<Tensor1D<M>, Axis<1>>::select(&result, &select_indices);
            // let mut result_slice: Tensor1D<M, H> = result.select(&select_indices);
            result_slice.mut_data() = self[n].mut_data();
            // <Cpu as DeviceReduce<_, Axis<1>>>::broadcast_into_no_reset::<CopyAccum>(result_slice.mut_data(), self[n].data());
        }

        // backward

        // // move tape out of all input tensors and merge it together
        // let phantom_out = out.clone();
        // let (lhs, lhs_tape) = lhs.split_tape();
        // let (rhs, rhs_tape) = rhs.split_tape();
        // // add backward op:
        // let mut tape = lhs_tape.merge(rhs_tape);
        // tape.add_backward_op(move |grads| f(lhs, rhs, phantom_out, grads));
        // out.put_tape(tape);
        //
        //
        //
        // move_tape_and_add_backward_op(self, result, move |t, result, grads| {
        //     let (t_grad, result_grad) = grads.mut_and_ref(&t, &result);
        //     <Cpu as DeviceReduce<_, $AxesTy>>::reduce_into_no_reset::<AddAccum>(t_grad, result_grad);
        // })
        move_tape_and_add_backward_listop(&self, result, move |t_list, result, grads| {
            for n in 0..N {
                let select_indices: [usize; M] = [n; M];
                let mut result_slice: Tensor1D<M> = SelectTo::<Tensor1D<M>, Axis<1>>::select(&result, &select_indices);
                let (mut t_grad, result_grad) = grads.mut_and_ref(&t_list[n], &result_slice);
                t_grad = result_grad;
            }
        })
    }
}

// TODO: I get `tensor_ops::select::SelectTo<structs::Tensor1D<M, H>, arrays::Axis<1>>` is not implemented for `structs::Tensor2D<M, N>``
// Is it because of the tape? doesn't work without tape?
// example
// ```
// let select_indices: [usize; M] = [n; M];
// let mut result_slice: Tensor1D<M, H> = SelectTo::<Tensor1D<M, H>, Axis<1>>::select(&result, &select_indices);
// ```
// Looks like it works when I remove H?

// TODO: also using Select consumes the original tensor, because we don't want multiple owners of the slice of data.
// How can I select multiple slices of data, without running ownership issue?

impl<const M: usize, const N: usize, H: Tape> UnstackTo<[Tensor1D<M, H>; N], Tensor1D<M, H>, Axis<1>> for Tensor2D<M, N, H> {
    fn unstack(self) -> [Tensor1D<M, H>; N] {
        let mut result: [Tensor1D<M, H>; N] = Default::default();
        for n in 0..N {
            let mut input_slice: Tensor1D<M, H> = self.select(&[n]);
            result[n].mut_data() = input_slice.mut_data();
            // <Cpu as DeviceReduce<_, Axis<1>>>::broadcast_into_no_reset::<CopyAccum>(result[n].mut_data(), input_slice.data());
        }
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::dbg;

    #[test]
    fn test_valid_1d_broadcasts() {
        let mut tensors: [Tensor1D<3>; 2] = [
            Tensor1D::zeros(),
            Tensor1D::zeros()
        ];
        let stacked = tensors.stack();
        // let stacked = Stack::stack(tensors);
        dbg!(stacked);
    }
}
