use crate::{
    shapes::{Dtype, Shape},
    tensor::{cpu::NdIndex, *},
};

use std::sync::Arc;

impl<E: Dtype> super::RollKernel<E> for Cpu {
    fn forward<S: Shape>(
        &self,
        op: super::RollOp,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let dims = inp.shape.concrete();
        let strides = inp.shape.strides();
        let mut data = self.try_alloc_zeros::<E>(inp.shape.num_elements())?;
        let mut idx = NdIndex::new(inp.shape, inp.strides);
        while let Some((old_i, mut idx)) = idx.next_with_idx() {
            idx[op.axis] = (idx[op.axis] + op.amount) % dims[op.axis];
            let new_i = idx
                .into_iter()
                .zip(strides)
                .map(|(i, s)| i * s)
                .sum::<usize>();
            data[new_i] = inp.data[old_i];
        }
        Ok(Tensor {
            id: unique_id(),
            data: Arc::new(data),
            shape: inp.shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        })
    }
    fn backward<S: Shape>(
        &self,
        op: super::RollOp,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let dims = inp.shape.concrete();
        let strides = inp.shape.strides();
        let mut idx = NdIndex::new(inp.shape, inp.strides);
        while let Some((old_i, mut idx)) = idx.next_with_idx() {
            idx[op.axis] = (idx[op.axis] + op.amount) % dims[op.axis];
            let new_i = idx
                .into_iter()
                .zip(strides)
                .map(|(i, s)| i * s)
                .sum::<usize>();
            grad_inp[old_i] += grad_out[new_i];
        }
        Ok(())
    }
}
