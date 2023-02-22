use super::visitors::{RecursiveWalker, TensorCollection, TensorOptions, TensorRef, VisitTensors};

use crate::{shapes::*, tensor::*};

use std::{string::String, vec::Vec};

struct Counter(usize);
impl<E: Dtype, D: DeviceStorage> VisitTensors<E, D> for Counter {
    type Container = TensorRef;
    type Err = D::Err;

    fn visit<S: Shape>(
        &mut self,
        _: String,
        opts: TensorOptions<S, E, D>,
        t: &Tensor<S, E, D>,
    ) -> Result<(), D::Err> {
        if opts.update {
            self.0 += t.shape().num_elements();
        }
        Ok(())
    }
}
pub trait NumParams<E: Dtype, D: DeviceStorage>: TensorCollection<E, D> {
    fn num_trainable_params(&self) -> usize {
        let mut op = Counter(0);
        Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut op,
            path: &mut Vec::new(),
        })
        .unwrap();
        op.0
    }
}
impl<E: Dtype, D: DeviceStorage, M: TensorCollection<E, D>> NumParams<E, D> for M {}
