use super::tensor_collection::*;

use crate::{shapes::*, tensor::*, tensor_ops::Device};

use std::vec::Vec;

/// Zero's any gradients associated with `self`.
///
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let model = dev.build_module::<Linear<2, 5>, f32>();
/// let mut grads: Gradients<f32, _> = model.alloc_grads();
/// model.zero_grads(&mut grads);
/// ```
pub trait ZeroGrads<E: Dtype, D: Device<E>>: TensorCollection<E, D> {
    /// Allocates gradients for this tensor collection. **This marks all other
    /// gradients as temporary, so they are dropped after .backward()**
    fn alloc_grads(&self) -> Gradients<E, D> {
        self.try_alloc_grads().unwrap()
    }

    /// Allocates gradients for this tensor collection. **This marks all other
    /// gradients as temporary, so they are dropped after .backward()**
    fn try_alloc_grads(&self) -> Result<Gradients<E, D>, D::Err> {
        // NOTE: try_zero_grads will add the leafs!
        let mut grads = Gradients::leaky();
        self.try_zero_grads(&mut grads)?;
        Ok(grads)
    }

    /// Zero's any gradients associated with `self`.
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
        })?;
        op.gradients.retain_leafs(&op.updated);
        Ok(())
    }
}
impl<E: Dtype, D: Device<E>, M: TensorCollection<E, D>> ZeroGrads<E, D> for M {}

struct ZeroGradOp<'a, E: Unit, D: DeviceStorage> {
    updated: Vec<UniqueId>,
    gradients: &'a mut Gradients<E, D>,
}

impl<'a, E: Dtype, D: Device<E>> TensorVisitor<E, D> for ZeroGradOp<'a, E, D> {
    type Viewer = ViewTensorRef;
    type Err = D::Err;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        t: &Tensor<S, E, D>,
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        if opts.do_gradient_update {
            let grad = self.gradients.get_or_alloc_mut(t)?;
            t.device.try_fill_with_zeros(grad)?;
            self.updated.push(t.id);
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nn::builders::{BatchNorm2D, DeviceBuildExt, Linear},
        tests::*,
    };

    use super::*;

    #[test]
    fn test_zero_grad() {
        let dev: TestDevice = Default::default();
        type Model = (Linear<2, 5>, BatchNorm2D<3>);
        let model = dev.build_module::<Model, TestDtype>();
        let mut grads: Gradients<TestDtype, TestDevice> = model.alloc_grads();

        let tmp1: Tensor<Rank1<5>, TestDtype, _> = dev.zeros();
        grads.get_or_alloc_mut(&tmp1).unwrap();

        let tmp2: Tensor<Rank1<5>, TestDtype, _> = dev.zeros();
        grads.get_or_alloc_mut(&tmp2).unwrap();

        model.zero_grads(&mut grads);
        assert_eq!(grads.get(&model.0.weight).array(), [[0.0; 2]; 5]);
        assert_eq!(grads.get(&model.0.bias).array(), [0.0; 5]);
        assert_eq!(grads.get(&model.1.scale).array(), [0.0; 3]);
        assert_eq!(grads.get(&model.1.bias).array(), [0.0; 3]);
        assert!(grads.get_ref_checked(&model.1.running_mean).is_none());
        assert!(grads.get_ref_checked(&model.1.running_var).is_none());
        assert!(grads.get_ref_checked(&tmp1).is_none());
        assert!(grads.get_ref_checked(&tmp2).is_none());
    }
}
