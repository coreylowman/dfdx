use super::tensor_collection::*;

use crate::{gradients::Gradients, shapes::*, tensor::*, unique_id::UniqueId};

use std::{string::String, vec::Vec};

/// Zero's any gradients associated with `self`.
pub trait ZeroGrads<E: Dtype, D: ZeroFillStorage<E>>: TensorCollection<E, D> {
    /// Zero's any gradients associated with `self`.
    ///
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let model = dev.build_module::<Linear<2, 5>, f32>();
    /// let mut grads: Gradients<f32, _> = Default::default();
    /// model.zero_grads(&mut grads);
    /// ```
    fn zero_grads(&self, gradients: &mut Gradients<E, D>) {
        self.try_zero_grads(gradients).unwrap();
    }

    /// Zero's any gradients associated with `self`.
    fn try_zero_grads(&self, gradients: &mut Gradients<E, D>) -> Result<(), D::Err> {
        let mut op = ZeroGradOp {
            updated: Vec::new(),
            gradients,
        };
        Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut op,
            path: &mut Vec::new(),
        })?;
        op.gradients.retain(&op.updated);
        Ok(())
    }
}
impl<E: Dtype, D: ZeroFillStorage<E>, M: TensorCollection<E, D>> ZeroGrads<E, D> for M {}

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
            let grad = self.gradients.get_or_alloc_mut(t)?;
            t.device.try_fill_with_zeros(grad)?;
            self.updated.push(t.id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nn::{
            builders::{BatchNorm2D, Linear},
            DeviceBuildExt,
        },
        tests::*,
    };

    use super::*;

    #[test]
    fn test_zero_grad() {
        let dev: TestDevice = Default::default();
        type Model = (Linear<2, 5>, BatchNorm2D<3>);
        let model = dev.build_module::<Model, TestDtype>();
        let mut grads = Default::default();
        model.zero_grads(&mut grads);
        assert_eq!(grads.get(&model.0.weight).array(), [[0.0; 2]; 5]);
        assert_eq!(grads.get(&model.0.bias).array(), [0.0; 5]);
        assert_eq!(grads.get(&model.1.scale).array(), [0.0; 3]);
        assert_eq!(grads.get(&model.1.bias).array(), [0.0; 3]);
        // assert!(grads.g)
    }
}