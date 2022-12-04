mod cpu_kernel;

use std::marker::PhantomData;

use crate::arrays::{Dtype, Shape};
use crate::gradients::Gradients;
use crate::tensor::{Cpu, DeviceStorage, Tensor};

use super::optimizer::*;

/// Configuration of hyperparameters for [Sgd].
///
/// Using different learning rate:
/// ```rust
/// # use dfdx::prelude::*;
/// SgdConfig {
///     lr: 1e-1,
///     momentum: None,
///     weight_decay: None,
/// };
/// ```
///
/// Using classic momentum:
/// ```rust
/// # use dfdx::prelude::*;
/// SgdConfig {
///     lr: 1e-2,
///     momentum: Some(Momentum::Classic(0.5)),
///     weight_decay: None,
/// };
/// ```
///
/// Using nesterov momentum:
/// ```rust
/// # use dfdx::prelude::*;
/// SgdConfig {
///     lr: 1e-3,
///     momentum: Some(Momentum::Nesterov(0.25)),
///     weight_decay: None,
/// };
/// ```
///
/// Using L2 weight decay:
/// ```rust
/// # use dfdx::prelude::*;
/// SgdConfig {
///     lr: 1e-3,
///     momentum: None,
///     weight_decay: Some(WeightDecay::L2(1e-2)),
/// };
/// ```
///
/// Using decoupled weight decay:
/// ```rust
/// # use dfdx::prelude::*;
/// SgdConfig {
///     lr: 1e-3,
///     momentum: None,
///     weight_decay: Some(WeightDecay::Decoupled(1e-2)),
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SgdConfig<E> {
    /// Learning rate. Defaults to `1e-2`
    pub lr: E,

    /// Optional momentum. Defaults to `None`.
    pub momentum: Option<Momentum<E>>,

    /// Optional weight decay. Defaults to `None`.
    pub weight_decay: Option<WeightDecay<E>>,
}

impl Default for SgdConfig<f32> {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            momentum: None,
            weight_decay: None,
        }
    }
}

/// Implementation of Stochastic Gradient Descent. Based on [pytorch's implementation](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
///
/// Nesterov Momentum is implemented as described in
/// [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html).
///
/// Weight decay is implemented as described in
/// [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
/// Both L2 weight_decay and decoupled weight_decay are available.
///
/// # Example Usage
///
/// Constructing using default:
/// ```rust
/// # use dfdx::prelude::*;
/// # type Model = Tensor0D;
/// let mut opt: Sgd<Model> = Default::default();
/// ```
///
/// Constructing using new:
/// ```rust
/// # use dfdx::prelude::*;
/// # type Model = Tensor0D;
/// let mut opt: Sgd<Model> = Sgd::new(SgdConfig {
///     lr: 1e-3,
///     momentum: Some(Momentum::Classic(0.5)),
///     weight_decay: Some(WeightDecay::L2(0.01)),
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug)]
pub struct Sgd<M, D: DeviceStorage = Cpu, E: Dtype = f32> {
    /// Hyperparameter configuration
    pub cfg: SgdConfig<E>,

    velocity: Gradients<D>,
    gradients: Gradients<D>,

    marker: PhantomData<*const M>,
}

impl<M, D: DeviceStorage, E: Dtype> Default for Sgd<M, D, E>
where
    SgdConfig<E>: Default,
{
    /// See [SgdConfig]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<M, D: DeviceStorage, E: Dtype> Sgd<M, D, E> {
    /// Constructs using hyperparameters from `cfg`
    pub fn new(cfg: SgdConfig<E>) -> Self {
        Self {
            cfg,
            velocity: Default::default(),
            gradients: Default::default(),
            marker: PhantomData,
        }
    }
}

pub(super) trait SgdKernel<E: Dtype>: DeviceStorage {
    fn update<S: Shape>(
        cfg: &SgdConfig<E>,
        param: &mut Self::Storage<S, E>,
        velocity: &mut Self::Storage<S, E>,
        grad: Self::Storage<S, E>,
    );
}

impl<M, D: DeviceStorage + SgdKernel<E>, E: Dtype> UpdateParams<D, E> for Sgd<M, D, E> {
    fn update_param<S: Shape>(
        &mut self,
        p: &mut Tensor<S, E, D>,
        unused: &mut super::optimizer::UnusedTensors,
    ) -> Result<(), <D>::Err> {
        let g = self.gradients.remove(p);
        match g {
            None => unused.add(p),
            Some(g) => {
                let v = self.velocity.get_or_alloc_mut(p)?;
                D::update(&self.cfg, &mut p.storage, v, g);
            }
        }
        Ok(())
    }
}

impl<E: Dtype, D: DeviceStorage, M: CanUpdateWithGradients<D, E>> Optimizer<M, D> for Sgd<M, D, E>
where
    Self: UpdateParams<D, E>,
{
    fn update(
        &mut self,
        module: &mut M,
        gradients: Gradients<D>,
    ) -> Result<(), OptimizerUpdateError<D>> {
        self.gradients = gradients;
        let mut unused = Default::default();
        match module.update(self, &mut unused) {
            Ok(_) => unused.into(),
            Err(e) => Err(OptimizerUpdateError::DeviceError(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_perfect_sgd() {
        let dev = build_test_device!();
        let mut sgd = Sgd::new(SgdConfig {
            lr: 1.0,
            momentum: None,
            weight_decay: None,
        });

        let mut pred: Tensor1D<5, _> = dev.zeros();
        let targ: Tensor1D<5, _> = dev.ones();
        for _ in 0..5 {
            let loss = (pred.trace() - targ.clone()).abs().mean();
            let gradients = loss.backward();
            sgd.update(&mut pred, gradients).expect("");
        }
        assert_close(&pred.array(), &[1.0; 5]);
        assert_close(&targ.array(), &[1.0; 5]);
    }

    #[test]
    fn test_sgd_no_momentum() {
        let dev = build_test_device!();
        let mut sgd = Sgd::new(Default::default());

        let mut t: Tensor1D<5, _> = dev.ones();
        let rate = dev.tensor([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9998, 0.998, 0.996, 0.98, 0.8],
            [0.99960005, 0.99600005, 0.992, 0.96000004, 0.6],
            [0.9994001, 0.9940001, 0.988, 0.94000006, 0.40000004],
            [0.9992001, 0.9920001, 0.98399997, 0.9200001, 0.20000005],
            [0.99900013, 0.9900001, 0.97999996, 0.9000001, 5.9604645e-8],
        ];

        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).mean().backward();
            sgd.update(&mut t, gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_sgd_classic_momentum() {
        let dev = build_test_device!();

        let mut sgd = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: Some(Momentum::Classic(0.5)),
            weight_decay: None,
        });

        let mut t: Tensor1D<5, _> = dev.ones();
        let rate = dev.tensor([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9998, 0.998, 0.996, 0.98, 0.8],
            [0.99950004, 0.995, 0.99, 0.95000005, 0.5],
            [0.99915004, 0.9915, 0.983, 0.915, 0.15],
            [0.99877506, 0.98775, 0.9755, 0.8775, -0.225],
            [0.9983876, 0.983875, 0.96775, 0.83875, -0.61249995],
        ];

        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).mean().backward();
            sgd.update(&mut t, gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_sgd_nesterov_momentum() {
        let dev = build_test_device!();

        let mut sgd = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: Some(Momentum::Nesterov(0.5)),
            weight_decay: None,
        });

        let mut t: Tensor1D<5, _> = dev.ones();
        let rate = dev.tensor([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9997, 0.997, 0.994, 0.97, 0.70000005],
            [0.99935, 0.9935, 0.987, 0.935, 0.35000005],
            [0.99897504, 0.98974997, 0.9795, 0.8975, -0.024999946],
            [0.99858755, 0.98587495, 0.97175, 0.85875, -0.41249993],
            [0.9981938, 0.98193747, 0.963875, 0.819375, -0.8062499],
        ];

        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).mean().backward();
            sgd.update(&mut t, gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_sgd_weight_decay_no_momentum() {
        let dev = build_test_device!();

        // With no momentum, both versions should be the same
        let mut sgd_l2 = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: None,
            weight_decay: Some(WeightDecay::L2(1e-1)),
        });
        let mut sgd_decoupled = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: None,
            weight_decay: Some(WeightDecay::Decoupled(1e-1)),
        });

        let mut t: Tensor1D<5, _> = dev.ones();
        let rate = dev.tensor([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9988, 0.997, 0.995, 0.979, 0.799],
            [0.99760115, 0.994003, 0.990005, 0.958021, 0.59820104],
            [0.9964036, 0.991009, 0.98501503, 0.937063, 0.39760286],
            [0.9952072, 0.988018, 0.98003, 0.9161259, 0.19720526],
            [0.994012, 0.98502994, 0.97505, 0.8952098, -0.00299193],
        ];
        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).mean().backward();
            sgd_l2.update(&mut t, gradients).expect("");
            assert_close(&t.array(), e);
        }
        t = dev.ones();
        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).mean().backward();
            sgd_decoupled.update(&mut t, gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_sgd_decoupled_weight_decay_classic_momentum() {
        let dev = build_test_device!();

        let mut sgd = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: Some(Momentum::Classic(0.5)),
            weight_decay: Some(WeightDecay::Decoupled(1e-1)),
        });

        let mut t: Tensor1D<5, _> = dev.ones();
        let rate = dev.tensor([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9988, 0.997, 0.995, 0.979, 0.799],
            [0.9975012, 0.993003, 0.988005, 0.948021, 0.498201],
            [0.9961537, 0.98851, 0.980017, 0.912073, 0.147703],
            [0.9947826, 0.983771, 0.971537, 0.873661, -0.227445],
            [0.9934003, 0.978913, 0.962815, 0.834037, -0.614717],
        ];
        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).mean().backward();
            sgd.update(&mut t, gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_sgd_l2_weight_decay_classic_momentum() {
        let dev = build_test_device!();

        // adding l2_weight_decay should be equivalent to adding an L2 term to the loss
        let weight_decay = 1e-1;
        let mut sgd_l2 = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: Some(Momentum::Classic(0.5)),
            weight_decay: Some(WeightDecay::L2(weight_decay)),
        });
        let mut sgd = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: Some(Momentum::Classic(0.5)),
            weight_decay: None,
        });

        let mut t: Tensor1D<5, _> = dev.ones();
        let rate = dev.tensor([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9988, 0.997, 0.995, 0.979, 0.799],
            [0.9970012, 0.992503, 0.987505, 0.947521, 0.49770102],
            [0.99490476, 0.987262, 0.97877, 0.91083395, 0.14655378],
            [0.99266165, 0.9816542, 0.9694238, 0.8715796, -0.22916639],
            [0.99034745, 0.9758687, 0.9597812, 0.83108085, -0.6167973],
        ];
        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).mean().backward();
            sgd_l2.update(&mut t, gradients).expect("");
            assert_close(&t.array(), e);
        }

        // Should be equivalent to l2 regularization, even with momentum
        t = dev.ones();
        for e in expected.iter() {
            let normal_loss = (t.trace() * rate.clone()).mean();
            let l2_loss = t.trace().powi(2).sum() * (weight_decay / (2.0));
            let loss = l2_loss + normal_loss;

            let gradients = loss.backward();
            sgd.update(&mut t, gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    // #[test]
    // fn test_sgd_changes_all_params() {
    //     type Model = (Linear<5, 16>, ReLU, Linear<16, 16>, ReLU, Linear<16, 10>);
    //     let mut rng = StdRng::seed_from_u64(0);
    //     let mut model: Model = Default::default();
    //     model.reset_params(&mut rng);
    //     let model_0 = model.clone();

    //     let x: Tensor2D<16, 5> = Tensor2D::rand(&mut rng);
    //     let y: Tensor2D<16, 10> = Tensor2D::rand(&mut rng);
    //     let mut opt: Sgd<Model> = Default::default();

    //     let py = model.forward(x.trace());
    //     let loss = (py - y).square().mean();
    //     let gradients = backward(loss);
    //     opt.update(&mut model, gradients).expect("");

    //     let model_1 = model.clone();

    //     assert!(model_0.0.weight.data() != model_1.0.weight.data());
    //     assert!(model_0.0.bias.data() != model_1.0.bias.data());
    //     assert!(model_0.2.weight.data() != model_1.2.weight.data());
    //     assert!(model_0.2.bias.data() != model_1.2.bias.data());
    //     assert!(model_0.4.weight.data() != model_1.4.weight.data());
    //     assert!(model_0.4.bias.data() != model_1.4.bias.data());
    // }

    // #[test]
    // fn test_sgd_unused_params() {
    //     type Model = (Linear<5, 16>, Linear<16, 10>);
    //     let mut model: Model = Default::default();
    //     let mut opt: Sgd<Model> = Default::default();
    //     let y = model.1.forward(Tensor2D::<8, 16>::zeros().trace());
    //     let g = backward(y.mean());
    //     opt.update(&mut model, g).expect_err("");
    // }
}
