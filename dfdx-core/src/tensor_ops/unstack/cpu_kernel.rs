use crate::{
    prelude::NoneTape,
    shapes::*,
    tensor::{unique_id, Cpu, Error, Tensor},
};

// note: in order to return NoneTape items and not require a tape type information T,
// each element must be optional.
impl<E: Dtype> super::UnstackKernel<E> for Cpu {
    fn forward<S: Shape, OptionalItems>(
        &self,
        stack: Tensor<S, E, Self, NoneTape>,
    ) -> Result<OptionalItems, Error>
    where
        S: super::SubDim,
        OptionalItems: Array<Option<Tensor<S::Tail, E, Self, NoneTape>>, Dim = S::Head>,
    {
        let (head, tail) = stack.shape().sub_dim();
        let stack_data = stack.data.as_slice();
        let unstack_num_elements = tail.num_elements();
        Ok(OptionalItems::from_fn(
            |i| {
                let mut data = self
                    .try_alloc_elem(unstack_num_elements, E::default())
                    // TODO: remove unwrap (needs try_from_fn)
                    // https://github.com/rust-lang/rust/issues/89379
                    .unwrap();

                data.copy_from_slice(
                    &stack_data[i * unstack_num_elements..(i + 1) * unstack_num_elements],
                );

                Some(Tensor {
                    id: unique_id(),
                    data: std::sync::Arc::new(data),
                    shape: *tail.shape(),
                    strides: tail.strides(),
                    device: self.clone(),
                    tape: NoneTape,
                })
            },
            head,
        ))
    }
    fn backward(
        &self,
        grad_stack: &mut Self::Vec,
        grad_unstack: &Self::Vec,
        unstack_idx: usize,
    ) -> Result<(), Error> {
        let unstack_num_elements = grad_unstack.len();
        for (i, stacked) in grad_stack
            .iter_mut()
            .skip(unstack_idx * unstack_num_elements)
            .take(unstack_num_elements)
            .enumerate()
        {
            *stacked += grad_unstack[i];
        }

        Ok(())
    }
}
