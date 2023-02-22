use super::tensor_collection::{
    RecursiveWalker, TensorCollection, TensorOptions, TensorVisitor, ViewTensorMut,
};

use crate::{shapes::*, tensor::*};

use std::{string::String, vec::Vec};

struct Resetter;
impl<E: Dtype, D: DeviceStorage> TensorVisitor<E, D> for Resetter {
    type Viewer = ViewTensorMut;
    type Err = D::Err;

    fn visit<S: Shape>(
        &mut self,
        _: String,
        opts: TensorOptions<S, E, D>,
        t: &mut Tensor<S, E, D>,
    ) -> Result<(), D::Err> {
        (opts.reset)(t)
    }
}
pub trait ResetParams<E: Dtype, D: DeviceStorage>: TensorCollection<E, D> {
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap();
    }
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut Resetter,
            path: &mut Vec::new(),
        })
    }
}
impl<E: Dtype, D: DeviceStorage, M: TensorCollection<E, D>> ResetParams<E, D> for M {}
