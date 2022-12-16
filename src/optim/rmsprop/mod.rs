mod cpu_kernel;

use std::marker::PhantomData;

use crate::{
    gradients::Gradients,
    shapes::{Dtype, Shape},
    tensor::{Cpu, DeviceStorage, OneFillStorage, Tensor},
};

use super::{
    GradientUpdate, Optimizer, OptimizerUpdateError, ParamUpdater, UnusedTensors, WeightDecay,
};

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

impl Default for RMSpropConfig<f32> {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
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
/// Constructing using default:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// # type Model = Tensor<Rank0>;
/// let mut opt: RMSprop<Model> = Default::default();
/// ```
///
/// Constructing using new:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// # type Model = Tensor<Rank0>;
/// let rmsprop: RMSprop<Model> = RMSprop::new(RMSpropConfig {
///     lr: 1e-3,
///     alpha: 0.5,
///     eps: 1e-8,
///     momentum: Some(0.5),
///     centered: false,
///     weight_decay: Some(WeightDecay::Decoupled(1e-1)),
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug)]
pub struct RMSprop<M, D: DeviceStorage = Cpu, E: Dtype = f32> {
    /// Hyperparameter configuration
    pub cfg: RMSpropConfig<E>,

    step: usize,
    momentums: Gradients<D>,
    square_avg: Gradients<D>,
    grad_avg: Gradients<D>,
    gradients: Gradients<D>,

    marker: PhantomData<*const M>,
}

impl<M, D: DeviceStorage, E: Dtype> Default for RMSprop<M, D, E>
where
    RMSpropConfig<E>: Default,
{
    /// See [RMSpropConfig]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<M, D: DeviceStorage, E: Dtype> RMSprop<M, D, E> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(cfg: RMSpropConfig<E>) -> Self {
        Self {
            cfg,
            step: 0,
            momentums: Default::default(),
            square_avg: Default::default(),
            grad_avg: Default::default(),
            gradients: Default::default(),
            marker: PhantomData,
        }
    }
}

pub(super) trait RMSpropKernel<E: Dtype>: DeviceStorage {
    fn update<S: Shape>(
        cfg: &RMSpropConfig<E>,
        param: &mut Self::Storage<S, E>,
        momentum: &mut Self::Storage<S, E>,
        square_avg: &mut Self::Storage<S, E>,
        grad_avg: &mut Self::Storage<S, E>,
        grad: Self::Storage<S, E>,
    );
}

impl<M, D: RMSpropKernel<f32> + OneFillStorage<f32>> ParamUpdater<D, f32> for RMSprop<M, D, f32> {
    fn update_param<S: Shape>(
        &mut self,
        p: &mut Tensor<S, f32, D>,
        unused: &mut UnusedTensors,
    ) -> Result<(), <D>::Err> {
        let g = self.gradients.remove(p);
        match g {
            None => unused.add(p),
            Some(g) => {
                let m = self.momentums.get_or_alloc_mut(p)?;
                let sa = self.square_avg.get_or_alloc_mut(p)?;
                let ga = self.grad_avg.get_or_alloc_mut(p)?;

                if self.step == 0 {
                    p.device.try_fill_with_ones(sa)?;
                }

                D::update(&self.cfg, &mut p.storage, m, sa, ga, g);
            }
        }
        Ok(())
    }
}

impl<E: Dtype, D: DeviceStorage, M: GradientUpdate<D, E>> Optimizer<M, D, E> for RMSprop<M, D, E>
where
    Self: ParamUpdater<D, E>,
{
    fn update(
        &mut self,
        module: &mut M,
        gradients: Gradients<D>,
    ) -> Result<(), OptimizerUpdateError<D>> {
        self.gradients = gradients;
        let mut unused = Default::default();
        let r = match module.update(self, &mut unused) {
            Ok(_) => unused.into(),
            Err(e) => Err(OptimizerUpdateError::DeviceError(e)),
        };
        self.step += 1;
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::build_test_device;

    fn test_matches_expected(cfg: RMSpropConfig<f32>, expected: [[f32; 5]; 5]) {
        let dev = build_test_device!();
        let rate = dev.tensor([0.1, 1.0, 2.0, 10.0, 100.0]);
        let mut t: Tensor1D<5, _> = dev.ones();
        let mut opt = RMSprop::new(cfg);
        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).square().sum().backward();
            opt.update(&mut t, gradients).expect("");
            assert_eq!(&t.array(), e);
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
        const EXPECTED: [[f32; 5]; 5] = [
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
        const EXPECTED: [[f32; 5]; 5] = [
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
        const EXPECTED: [[f32; 5]; 5] = [
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
        const EXPECTED: [[f32; 5]; 5] = [
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
        const EXPECTED: [[f32; 5]; 5] = [
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
        const EXPECTED: [[f32; 5]; 5] = [
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
        const EXPECTED: [[f32; 5]; 5] = [
            [0.9947892, 0.97745883, 0.9653907, 0.9633808, 0.96337724],
            [0.98959416, 0.9568803, 0.9387497, 0.93603325, 0.9360285],
            [0.9844144, 0.93768346, 0.91579914, 0.9127129, 0.9127075],
            [0.9792493, 0.91952574, 0.8950769, 0.89176315, 0.8917574],
            [0.9740982, 0.90218556, 0.8758817, 0.8724158, 0.87240976],
        ];
        test_matches_expected(cfg, EXPECTED);
    }

    // #[test]
    // fn test_rmsprop_unused_params() {
    //     type Model = (Linear<5, 16>, Linear<16, 10>);
    //     let mut model: Model = Default::default();
    //     let mut opt: RMSprop<Model> = Default::default();
    //     let y = model.1.forward(Tensor2D::<8, 16>::zeros().trace());
    //     let g = backward(y.mean());
    //     opt.update(&mut model, g).expect_err("");
    // }
}
