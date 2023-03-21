use num_traits::AsPrimitive;
use std::sync::Arc;

use crate::prelude::{Cpu, Shape, Tensor, Unit};

impl<E1: Unit + AsPrimitive<E2>, E2: Unit> super::ToDtypeKernel<E1, E2> for Cpu {
    fn forward<S: Shape>(inp: Tensor<S, E1, Self>) -> Result<Tensor<S, E2, Self>, Self::Err> {
        Ok(Tensor {
            id: crate::prelude::unique_id(),
            data: Arc::new(inp.data.iter().map(|x| (*x).as_()).collect()),
            shape: inp.shape,
            strides: inp.strides,
            device: inp.device.clone(),
            tape: inp.tape,
        })
    }
}
