use super::tensor_collection::*;

use crate::{prelude::Device, shapes::*, tensor::*};

struct ModelEMAOp<E> {
    decay: E,
}
impl<E: Dtype, D: Device<E>> TensorVisitor<E, D> for ModelEMAOp<E> {
    type Viewer = (ViewTensorMut, ViewTensorRef);
    type Err = D::Err;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        (dst, src): (&mut Tensor<S, E, D>, &Tensor<S, E, D>),
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        if opts.do_gradient_update {
            dst.try_axpy(self.decay, src, E::ONE - self.decay)?;
        }
        Ok(None)
    }
}

/// Performs model exponential moving average on two modules.
///
/// **Only updates trainable parameters**. For example, batch normalization
/// running parameters are not updated.
///
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = Linear<2, 5>;
/// let model = dev.build_module::<Model, f32>();
/// let mut model_ema = model.clone();
/// model_ema.ema(&model, 0.001);
/// ```
pub trait ModelEMA<E: Dtype, D: Device<E>>: TensorCollection<E, D> {
    /// Does `self = self * decay + other * (1 - decay), using
    /// [crate::tensor_ops::axpy()] on parameters.
    ///
    /// **Only updates trainable parameters**. For example, batch normalization
    /// running parameters are not updated.
    fn ema(&mut self, other: &Self, decay: E) {
        self.try_ema(other, decay).unwrap();
    }

    fn try_ema(&mut self, other: &Self, decay: E) -> Result<(), D::Err> {
        let mut op = ModelEMAOp { decay };
        Self::iter_tensors(&mut RecursiveWalker {
            m: (self, other),
            f: &mut op,
        })?;
        Ok(())
    }
}
impl<E: Dtype, D: Device<E>, M: TensorCollection<E, D>> ModelEMA<E, D> for M {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::builders::*, tensor_ops::axpy, tests::*};

    #[test]
    fn test_model_ema() {
        let dev: TestDevice = Default::default();
        let distr = rand_distr::Standard;

        type Model = (Linear<3, 5>, (Linear<5, 10>, BatchNorm2D<3>));
        let model = dev.build_module::<Model, TestDtype>();

        let mut ema1 = dev.build_module::<Model, TestDtype>();
        ema1.1 .1.running_mean.fill_with_distr(distr);
        ema1.1 .1.running_var.fill_with_distr(distr);
        let ema0 = ema1.clone();

        let decay: TestDtype = 0.5;

        ema1.ema(&model, decay);
        // check that batchnorm running params aren't updated
        {
            assert_eq!(
                ema1.1 .1.running_mean.array(),
                ema0.1 .1.running_mean.array()
            );
            assert_eq!(ema1.1 .1.running_var.array(), ema0.1 .1.running_var.array());
        }

        {
            assert_eq!(
                axpy(&ema0.0.weight, decay, &model.0.weight, 1.0 - decay).array(),
                ema1.0.weight.array()
            );
            assert_eq!(
                axpy(&ema0.0.bias, decay, &model.0.bias, 1.0 - decay).array(),
                ema1.0.bias.array()
            );
        }

        {
            assert_eq!(
                axpy(&ema0.1 .0.weight, decay, &model.1 .0.weight, 1.0 - decay).array(),
                ema1.1 .0.weight.array()
            );
            assert_eq!(
                axpy(&ema0.1 .0.bias, decay, &model.1 .0.bias, 1.0 - decay).array(),
                ema1.1 .0.bias.array()
            );
        }

        {
            assert_eq!(
                axpy(&ema0.1 .1.scale, decay, &model.1 .1.scale, 1.0 - decay).array(),
                ema1.1 .1.scale.array()
            );
            assert_eq!(
                axpy(&ema0.1 .1.bias, decay, &model.1 .1.bias, 1.0 - decay).array(),
                ema1.1 .1.bias.array()
            );
        }
    }
}
