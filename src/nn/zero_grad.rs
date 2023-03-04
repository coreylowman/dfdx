use super::tensor_collection::{
    RecursiveWalker, TensorCollection, TensorOptions, TensorVisitor, ViewTensorRef,
};

use crate::{gradients::Gradients, shapes::*, tensor::*, unique_id::UniqueId};

use std::{string::String, vec::Vec};

struct ZeroGradOp<'a, E: Unit, D: DeviceStorage> {
    updated: Vec<UniqueId>,
    gradients: &'a mut Gradients<E, D>,
}

impl<'a, E: Dtype, D: ZeroFillStorage<E>> TensorVisitor<E, D> for ZeroGradOp<'a, E, D> {
    type Viewer = ViewTensorRef;
    type Err = D::Err;

    fn visit<S: Shape>(
        &mut self,
        _: String,
        opts: TensorOptions<S, E, D>,
        t: &Tensor<S, E, D>,
    ) -> Result<(), Self::Err> {
        if opts.do_gradient_update {
            t.device.try_fill_with_zeros(self.gradients.get_mut(t))?;
            self.updated.push(t.id);
        }
        Ok(())
    }
}

pub trait ZeroGrad<E: Dtype, D: ZeroFillStorage<E>>: TensorCollection<E, D> {
    fn zero_grad(&self, gradients: &mut Gradients<E, D>) {
        self.try_zero_grad(gradients).unwrap();
    }

    fn try_zero_grad(&self, gradients: &mut Gradients<E, D>) -> Result<(), D::Err> {
        let mut op = ZeroGradOp {
            updated: Vec::new(),
            gradients,
        };
        Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut op,
            path: &mut Vec::new(),
        })?;
        todo!("Remove temporary gradients")
    }
}
impl<E: Dtype, D: ZeroFillStorage<E>, M: TensorCollection<E, D>> ZeroGrad<E, D> for M {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_grad() {
        todo!();
    }
}
