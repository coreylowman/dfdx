use super::tensor_collection::*;

use crate::{prelude::Device, shapes::*, tensor::*};

struct Counter(usize);
impl<E: Dtype, D: Device<E>> TensorVisitor<E, D> for Counter {
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
            self.0 += t.shape().num_elements();
        }
        Ok(None)
    }
}

/// Get the number of trainable parameters in a model.
///
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = Linear<2, 5>;
/// let model = dev.build_module::<Model, f32>();
/// assert_eq!(model.num_trainable_params(), 2 * 5 + 5);
/// ```
pub trait NumParams<E: Dtype, D: Device<E>>: TensorCollection<E, D> {
    /// Returns the number of trainable params in any model.
    fn num_trainable_params(&self) -> usize {
        let mut op = Counter(0);
        Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut op,
        })
        .unwrap();
        op.0
    }
}
impl<E: Dtype, D: Device<E>, M: TensorCollection<E, D>> NumParams<E, D> for M {}
