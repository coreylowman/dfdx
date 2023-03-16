mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use std::{marker::PhantomData, sync::Arc};

use crate::{
    nn::tensor_collection::*,
    shapes::{Dtype, Shape},
    tensor::*,
    tensor_ops::Device,
};

use super::{Optimizer, OptimizerUpdateError, UnusedTensors, WeightDecay};

/// Configuration of hyperparameters for [RMSprop].
#[derive(Debug, Clone, Copy)]
pub struct RMSpropConfig<E> {
    /// Learning rate. Defaults to `1e-2`.
    pub lr: E,

    /// Value for exponential moving average. Defaults to `0.9`.
    pub alpha: E,

    /// Epsilon for stability. Defaults to `1e-8`.
    pub eps: E,

    /// Optional momentum. Defaults to `None`.
    pub momentum: Option<E>,

    /// Whether the avg should be centered by the grad's avg value.
    /// Defaults to `false`.
    pub centered: bool,

    /// Optional weight decay. Defaults to `None`.
    pub weight_decay: Option<WeightDecay<E>>,
}

impl<E: Dtype> Default for RMSpropConfig<E> {
    fn default() -> Self {
        Self {
            lr: E::from_f32(1e-2).unwrap(),
            alpha: E::from_f32(0.9).unwrap(),
            eps: E::from_f32(1e-8).unwrap(),
            momentum: None,
            centered: false,
            weight_decay: None,
        }
    }
}

/// RMSprop As described in [Hinton, 2012](http://www.cs.toronto.edu/%7Etijmen/csc321/slides/lecture_slides_lec6.pdf).
///
/// This implementation is based off of RMSprop from
/// [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/rmsprop_tf.py)
/// because the pytorch implementation has [some issues](https://github.com/pytorch/pytorch/issues/23796).
///
/// The main difference between the pytorch implementation is that [RMSpropConfig::eps] is added inside of the sqrt()
/// operation.
///
/// The `lr_in_momentum` option is not provided because it didn't seem to make a difference in testing.
///
/// # Example Usage
///
/// Constructing using new:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// # type Model = Tensor<Rank0, f32, Cpu>;
/// # let dev: Cpu = Default::default();
/// # let model: Model = dev.zeros();
/// let rmsprop: RMSprop<Model, f32, Cpu> = RMSprop::new(&model, RMSpropConfig {
///     lr: 1e-3,
///     alpha: 0.5,
///     eps: 1e-8,
///     momentum: Some(0.5),
///     centered: false,
///     weight_decay: Some(WeightDecay::Decoupled(1e-1)),
/// });
#[derive(Debug)]
pub struct RMSprop<M, E: Dtype, D: DeviceStorage> {
    /// Hyperparameter configuration
    pub cfg: RMSpropConfig<E>,

    step: usize,
    momentums: Gradients<E, D>,
    square_avg: Gradients<E, D>,
    grad_avg: Gradients<E, D>,

    marker: PhantomData<*const M>,
}

impl<M, E: Dtype, D: DeviceStorage> RMSprop<M, E, D> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(_model: &M, cfg: RMSpropConfig<E>) -> Self {
        Self {
            cfg,
            step: 0,
            momentums: Gradients::leaky(),
            square_avg: Gradients::leaky(),
            grad_avg: Gradients::leaky(),
            marker: PhantomData,
        }
    }
}

pub trait RMSpropKernel<E: Dtype>: DeviceStorage {
    fn update(
        &self,
        cfg: &RMSpropConfig<E>,
        param: &mut Self::Vec<E>,
        momentum: &mut Self::Vec<E>,
        square_avg: &mut Self::Vec<E>,
        grad_avg: &mut Self::Vec<E>,
        grad: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

impl<M, E: Dtype, D: Device<E>> TensorVisitor<E, D>
    for (&mut RMSprop<M, E, D>, &Gradients<E, D>, UnusedTensors)
{
    type Viewer = ViewTensorMut;
    type Err = D::Err;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        p: &mut Tensor<S, E, D>,
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        if !opts.do_gradient_update {
            return Ok(None);
        }
        let g = self.1.get_ref_checked(p);
        match g {
            None => self.2.add(p),
            Some(g) => {
                let m = self.0.momentums.get_or_alloc_mut(p)?;
                let sa = self.0.square_avg.get_or_alloc_mut(p)?;
                let ga = self.0.grad_avg.get_or_alloc_mut(p)?;

                if self.0.step == 0 {
                    p.device.try_fill_with_ones(sa)?;
                }

                RMSpropKernel::update(
                    &p.device,
                    &self.0.cfg,
                    Arc::make_mut(&mut p.data),
                    m,
                    sa,
                    ga,
                    g,
                )?;
            }
        }
        Ok(None)
    }
}

impl<M: TensorCollection<E, D>, D: Device<E> + OneFillStorage<E>, E: Dtype> Optimizer<M, D, E>
    for RMSprop<M, E, D>
{
    fn update(
        &mut self,
        module: &mut M,
        gradients: &Gradients<E, D>,
    ) -> Result<(), OptimizerUpdateError<D>> {
        let mut op = (self, gradients, Default::default());
        let result = M::iter_tensors(&mut RecursiveWalker {
            m: module,
            f: &mut op,
        });
        let r = match result {
            Ok(_) => op.2.into(),
            Err(e) => Err(OptimizerUpdateError::DeviceError(e)),
        };
        op.0.step += 1;
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::*, tensor_ops::*, tests::*};

    fn test_matches_expected(cfg: RMSpropConfig<TestDtype>, expected: [[TestDtype; 5]; 5]) {
        let dev: TestDevice = Default::default();
        let rate: Tensor<_, TestDtype, _> = dev.tensor([0.1, 1.0, 2.0, 10.0, 100.0]);
        let mut t: Tensor<Rank1<5>, TestDtype, _> = dev.ones();
        let mut opt = RMSprop::new(&t, cfg);
        for e in expected.iter() {
            let gradients = (t.leaky_trace() * rate.clone()).square().sum().backward();
            opt.update(&mut t, &gradients).expect("");
            std::println!("{:?}", t.array());
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_rmsprop_default() {
        let cfg = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
            momentum: None,
            centered: false,
            weight_decay: None,
        };
        const EXPECTED: [[TestDtype; 5]; 5] = [
            [0.9997892, 0.98245883, 0.9703907, 0.9683808, 0.96837723],
            [0.99956703, 0.96670717, 0.9485176, 0.9457928, 0.945788],
            [0.9993329, 0.9521923, 0.9301649, 0.9270585, 0.9270531],
            [0.9990862, 0.9385879, 0.9138966, 0.9105493, 0.91054344],
            [0.9988262, 0.9256831, 0.8990271, 0.8955128, 0.8955067],
        ];
        test_matches_expected(cfg, EXPECTED);
    }

    #[test]
    fn test_rmsprop_momentum() {
        let cfg = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
            momentum: Some(0.9),
            centered: false,
            weight_decay: None,
        };
        const EXPECTED: [[TestDtype; 5]; 5] = [
            [0.9997892, 0.98245883, 0.9703907, 0.9683808, 0.96837723],
            [0.9993773, 0.9509201, 0.9218692, 0.9173355, 0.9173275],
            [0.9987725, 0.9082085, 0.86019397, 0.8530321, 0.8530196],
            [0.9979816, 0.8566451, 0.78923434, 0.7795164, 0.7794995],
            [0.9970101, 0.798177, 0.71185935, 0.69974965, 0.6997286],
        ];
        test_matches_expected(cfg, EXPECTED);
    }

    #[test]
    fn test_rmsprop_diff_alpha() {
        let cfg = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.5,
            eps: 1e-8,
            momentum: None,
            centered: false,
            weight_decay: None,
        };
        const EXPECTED: [[TestDtype; 5]; 5] = [
            [0.99971724, 0.9873509, 0.9859671, 0.985858, 0.98585784],
            [0.9993176, 0.9763115, 0.97450525, 0.97436625, 0.97436607],
            [0.9987531, 0.96588355, 0.9639029, 0.9637519, 0.96375173],
            [0.99795645, 0.95572895, 0.95366806, 0.95351166, 0.9535115],
            [0.99683434, 0.9457051, 0.9436056, 0.9434466, 0.9434464],
        ];
        test_matches_expected(cfg, EXPECTED);
    }

    #[test]
    fn test_rmsprop_diff_eps() {
        let cfg = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-2,
            momentum: None,
            centered: false,
            weight_decay: None,
        };
        const EXPECTED: [[TestDtype; 5]; 5] = [
            [0.9997904, 0.98252594, 0.97041094, 0.9683808, 0.96837723],
            [0.99956954, 0.9668238, 0.9485463, 0.94579285, 0.945788],
            [0.999337, 0.95234853, 0.93019867, 0.9270586, 0.9270531],
            [0.99909216, 0.9387773, 0.9139341, 0.91054934, 0.91054344],
            [0.9988343, 0.9259014, 0.89906746, 0.8955129, 0.8955067],
        ];
        test_matches_expected(cfg, EXPECTED);
    }

    #[test]
    fn test_rmsprop_centered() {
        let cfg = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
            momentum: None,
            centered: true,
            weight_decay: None,
        };
        const EXPECTED: [[TestDtype; 5]; 5] = [
            [0.9997892, 0.98218256, 0.96900064, 0.9666708, 0.9666667],
            [0.99956703, 0.965664, 0.9448974, 0.941596, 0.9415902],
            [0.9993329, 0.9498438, 0.9236177, 0.91970736, 0.91970056],
            [0.9990862, 0.93438274, 0.90377975, 0.89941716, 0.8994096],
            [0.9988262, 0.9190646, 0.8847198, 0.87998855, 0.8799804],
        ];
        test_matches_expected(cfg, EXPECTED);
    }

    #[test]
    fn test_rmsprop_l2_weight_decay() {
        let cfg = RMSpropConfig {
            weight_decay: Some(WeightDecay::L2(0.5)),
            ..Default::default()
        };
        const EXPECTED: [[TestDtype; 5]; 5] = [
            [0.9945992, 0.9797556, 0.97018003, 0.96838075, 0.96837723],
            [0.9890257, 0.96231526, 0.9482287, 0.94579273, 0.945788],
            [0.98328084, 0.94663495, 0.92983353, 0.92705846, 0.9270531],
            [0.9773666, 0.9321751, 0.9135383, 0.9105492, 0.91054344],
            [0.97128564, 0.9186157, 0.89865, 0.89551276, 0.8955067],
        ];
        test_matches_expected(cfg, EXPECTED);
    }

    #[test]
    fn test_rmsprop_decoupled_weight_decay() {
        let cfg = RMSpropConfig {
            weight_decay: Some(WeightDecay::Decoupled(0.5)),
            ..Default::default()
        };
        const EXPECTED: [[TestDtype; 5]; 5] = [
            [0.9947892, 0.97745883, 0.9653907, 0.9633808, 0.96337724],
            [0.98959416, 0.9568803, 0.9387497, 0.93603325, 0.9360285],
            [0.9844144, 0.93768346, 0.91579914, 0.9127129, 0.9127075],
            [0.9792493, 0.91952574, 0.8950769, 0.89176315, 0.8917574],
            [0.9740982, 0.90218556, 0.8758817, 0.8724158, 0.87240976],
        ];
        test_matches_expected(cfg, EXPECTED);
    }

    #[test]
    fn test_unused_tensors() {
        let dev: TestDevice = Default::default();
        let mut t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let mut opt = RMSprop::new(&t, Default::default());
        opt.update(&mut t, &Gradients::leaky()).expect_err("");
    }
}
