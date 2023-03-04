use super::tensor_collection::{
    RecursiveWalker, TensorCollection, TensorOptions, TensorVisitor, ViewTensorMut, ViewTensorRef,
};

use crate::{shapes::*, tensor::*, tensor_ops::axpy::AxpyKernel};

use std::{string::String, vec::Vec};

struct ModelEMAOp<E> {
    decay: E,
}
impl<E: Dtype, D: AxpyKernel<E>> TensorVisitor<E, D> for ModelEMAOp<E> {
    type Viewer = (ViewTensorMut, ViewTensorRef);
    type Err = D::Err;

    fn visit<S: Shape>(
        &mut self,
        _: String,
        opts: TensorOptions<S, E, D>,
        (dst, src): (&mut Tensor<S, E, D>, &Tensor<S, E, D>),
    ) -> Result<(), Self::Err> {
        if opts.do_gradient_update {
            dst.try_axpy(self.decay, src, E::ONE - self.decay)?;
        }
        Ok(())
    }
}

/// Performs model exponential moving average on two modules.
pub trait ModelEMA<E: Dtype, D: AxpyKernel<E>>: TensorCollection<E, D> {
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
            path: &mut Vec::new(),
        })
    }
}
impl<E: Dtype, D: AxpyKernel<E>, M: TensorCollection<E, D>> ModelEMA<E, D> for M {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{builders::*, DeviceBuildExt},
        tensor_ops::axpy,
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
