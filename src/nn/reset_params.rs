use super::tensor_collection::*;

use crate::{shapes::*, tensor::*};

struct Resetter;
impl<E: Dtype, D: DeviceStorage> TensorVisitor<E, D> for Resetter {
    type Viewer = ViewTensorMut;
    type Err = D::Err;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        t: &mut Tensor<S, E, D>,
    ) -> Result<(), D::Err> {
        (opts.reset)(t)
    }
}

/// Reset a module's parameters with their default reset function:
///
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = Linear<2, 5>;
/// let mut model = dev.build_module::<Model, f32>();
/// model.reset_params();
/// ```
pub trait ResetParams<E: Dtype, D: DeviceStorage>: TensorCollection<E, D> {
    /// Reset all a model's parameters.
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap();
    }
    /// Reset all a model's parameters.
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut Resetter,
        })
    }
}
impl<E: Dtype, D: DeviceStorage, M: TensorCollection<E, D>> ResetParams<E, D> for M {}
