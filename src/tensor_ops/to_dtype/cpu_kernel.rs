use num_traits::AsPrimitive;
use std::{sync::Arc, vec::Vec};

use crate::prelude::{cpu::CachableVec, Cpu, Shape, Tensor, Unit};

impl<E1: Unit + AsPrimitive<E2>, E2: Unit> super::ToDtypeKernel<E1, E2> for Cpu {
    fn forward<S: Shape>(inp: Tensor<S, E1, Self>) -> Result<Tensor<S, E2, Self>, Self::Err> {
        let data: &[E1] = inp.data.as_ref();
        let data: Vec<E2> = data.iter().map(|x| (*x).as_()).collect();
        let data = CachableVec {
            data,
            cache: inp.device.cache.clone(),
        };

        Ok(Tensor {
            id: crate::prelude::unique_id(),
            data: Arc::new(data),
            shape: inp.shape,
            strides: inp.strides,
            device: inp.device.clone(),
            tape: inp.tape,
        })
    }
}
