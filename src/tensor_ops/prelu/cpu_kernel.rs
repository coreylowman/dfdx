extern crate alloc;
use alloc::sync::Arc;

use crate::prelude::{Cpu, Tape, Shape, Dtype, Tensor, HasErr};

use super::PReLUDev;

impl<S:Shape, E: Dtype, T: Tape<E, Cpu>> PReLUDev<E, Cpu> for Tensor<S, E, Cpu, T> {
    fn try_prelu(self, a: E) -> Result<Self, <Cpu as HasErr>::Err> {
        let v = self.data.iter().map(|x| {
            if x>&E::from_f32(0.0).unwrap() {
                x.clone()
            }
            else {
                x.clone()*a.clone()
            }
        }).collect::<Vec<E>>();
        let t = Tensor {
            id: self.id,
            data: Arc::new(v),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            device: Cpu::default(),
            tape: T::default(),
        };
        Ok(t)
    }
}