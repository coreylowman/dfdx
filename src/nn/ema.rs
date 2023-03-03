use super::tensor_collection::{
    RecursiveWalker, TensorCollection, TensorOptions, TensorVisitor, ViewTensorMut, ViewTensorRef,
};

use crate::{shapes::*, tensor::*, tensor_ops::ema::EmaKernel};

use std::{string::String, vec::Vec};

struct ModelEMAOp<E> {
    decay: E,
}
impl<E: Dtype, D: EmaKernel<E>> TensorVisitor<E, D> for ModelEMAOp<E> {
    type Viewer = (ViewTensorMut, ViewTensorRef);
    type Err = D::Err;

    fn visit<S: Shape>(
        &mut self,
        _: String,
        opts: TensorOptions<S, E, D>,
        (dst, src): (&mut Tensor<S, E, D>, &Tensor<S, E, D>),
    ) -> Result<(), Self::Err> {
        if opts.do_gradient_update {
            dst.try_ema_assign(src, self.decay)?;
        }
        Ok(())
    }
}

/// Performs model exponential moving average on two modules.
pub trait ModelEMA<E: Dtype, D: EmaKernel<E>>: TensorCollection<E, D> {
    /// Does `self = self * decay + other * (1 - decay), using
    /// [crate::tensor_ops::ema] on parameters.
    ///
    /// **Only updates trainable parameters**. For example, batch normalization
    /// running parameters are not updated.
    fn ema_assign(&mut self, other: &Self, decay: E) {
        self.try_ema_assign(other, decay).unwrap();
    }

    fn try_ema_assign(&mut self, other: &Self, decay: E) -> Result<(), D::Err> {
        let mut op = ModelEMAOp { decay };
        Self::iter_tensors(&mut RecursiveWalker {
            m: (self, other),
            f: &mut op,
            path: &mut Vec::new(),
        })
    }
}
impl<E: Dtype, D: EmaKernel<E>, M: TensorCollection<E, D>> ModelEMA<E, D> for M {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{builders::*, DeviceBuildExt},
        tests::*,
    };

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

        ema1.ema_assign(&model, decay);
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
                ema0.0.weight.ema(&model.0.weight, decay).array(),
                ema1.0.weight.array()
            );
            assert_eq!(
                ema0.0.bias.ema(&model.0.bias, decay).array(),
                ema1.0.bias.array()
            );
        }

        {
            assert_eq!(
                ema0.1 .0.weight.ema(&model.1 .0.weight, decay).array(),
                ema1.1 .0.weight.array()
            );
            assert_eq!(
                ema0.1 .0.bias.ema(&model.1 .0.bias, decay).array(),
                ema1.1 .0.bias.array()
            );
        }

        {
            assert_eq!(
                ema0.1 .1.scale.ema(&model.1 .1.scale, decay).array(),
                ema1.1 .1.scale.array()
            );
            assert_eq!(
                ema0.1 .1.bias.ema(&model.1 .1.bias, decay).array(),
                ema1.1 .1.bias.array()
            );
        }
    }
}
