use core::marker::PhantomData;

use crate::{
    arrays::{Dtype, Shape},
    devices::DeviceStorage,
    gradients::Gradients,
};

use super::{CanUpdateWithGradients, Optimizer, OptimizerUpdateError, ParamUpdater, WeightDecay};

mod cpu;

/// Configuration of hyperparameters for [Adam].
///
/// Changing all default parameters:
/// ```rust
/// # use dfdx::prelude::*;
/// AdamConfig {
///     lr: 1e-2,
///     betas: [0.1, 0.2],
///     eps: 1e-6,
///     weight_decay: Some(WeightDecay::L2(1e-1)),
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig<E> {
    /// Learning rate. Defaults to `1e-3`.
    pub lr: E,

    /// Betas from Adam paper. Defaults to `[0.9, 0.999]`.
    pub betas: [E; 2],

    /// Epsilon for numerical stability. Defaults to `1e-8`.
    pub eps: E,

    /// Optional weight decay. Defaults to `None`.
    pub weight_decay: Option<WeightDecay<E>>,
}

impl Default for AdamConfig<f32> {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: None,
        }
    }
}

/// An implementation of the Adam optimizer from
/// [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
///
/// # Example Usage
///
/// Constructing using default:
/// ```rust
/// # use dfdx::prelude::*;
/// # type Model = Tensor0D;
/// let mut opt: Adam<Model> = Default::default();
/// ```
///
/// Changing using new
/// ```rust
/// # use dfdx::prelude::*;
/// # type Model = Tensor0D;
/// let mut opt: Adam<Model> = Adam::new(AdamConfig {
///     lr: 1e-2,
///     betas: [0.5, 0.25],
///     eps: 1e-6,
///     weight_decay: Some(WeightDecay::Decoupled(1e-2)),
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug)]
pub struct Adam<M, D: DeviceStorage, E: Dtype> {
    /// Hyperparameter configuration
    pub cfg: AdamConfig<E>,

    t: i32,
    gradients: Gradients<D>,
    moment1: Gradients<D>,
    moment2: Gradients<D>,

    marker: PhantomData<*const M>,
}

impl<M, D: DeviceStorage, E: Dtype> Default for Adam<M, D, E>
where
    AdamConfig<E>: Default,
{
    /// See [AdamConfig]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<M, D: DeviceStorage, E: Dtype> Adam<M, D, E> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(cfg: AdamConfig<E>) -> Self {
        Self {
            cfg,
            t: 0,
            gradients: Default::default(),
            moment1: Default::default(),
            moment2: Default::default(),
            marker: PhantomData,
        }
    }
}

pub(super) trait AdamUpdate<D: DeviceStorage, E: Dtype> {
    fn update_param<S: Shape>(
        &self,
        t: i32,
        param: &mut D::Storage<S, E>,
        moment1: &mut D::Storage<S, E>,
        moment2: &mut D::Storage<S, E>,
        grad: D::Storage<S, E>,
    );
}

impl<M, D: DeviceStorage, E: Dtype> ParamUpdater<D, E> for Adam<M, D, E>
where
    AdamConfig<E>: AdamUpdate<D, E>,
{
    fn update_param<S: Shape>(
        &mut self,
        p: &mut crate::tensor::Tensor<S, E, D>,
        unused: &mut super::UnusedTensors,
    ) -> Result<(), <D>::Err> {
        let g = self.gradients.remove(p);
        match g {
            None => unused.add(p),
            Some(g) => {
                let m_t = self.moment1.get_mut(p)?;
                let v_t = self.moment2.get_mut(p)?;
                self.cfg.update_param(self.t, &mut p.storage, m_t, v_t, g);
            }
        }
        Ok(())
    }
}

impl<E: Dtype, D: DeviceStorage, M: CanUpdateWithGradients<D, E>> Optimizer<M, D> for Adam<M, D, E>
where
    Self: ParamUpdater<D, E>,
{
    fn update(
        &mut self,
        module: &mut M,
        gradients: Gradients<D>,
    ) -> Result<(), OptimizerUpdateError<D>> {
        self.t = self.t.checked_add(1).unwrap();
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
    use crate::devices::*;
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_default_adam_params() {
        let dev = build_test_device!();
        let mut opt = Adam::default();
        let mut t: Tensor1D<5, _> = dev.ones();
        let rate = dev.tensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]);
        let expected = [
            [0.99999994, 0.999996, 0.9997143, 0.9990244, 0.99900025],
            [0.9999999, 0.999992, 0.99942863, 0.99804884, 0.9980005],
            [0.9999998, 0.999988, 0.999143, 0.9970733, 0.9970008],
            [0.99999976, 0.999984, 0.9988574, 0.99609786, 0.9960012],
            [0.9999997, 0.99998003, 0.9985718, 0.9951225, 0.9950017],
            [0.99999964, 0.99997604, 0.99828625, 0.99414724, 0.9940022],
            [0.9999996, 0.99997205, 0.99800074, 0.9931721, 0.9930029],
            [0.9999995, 0.99996805, 0.9977153, 0.9921971, 0.9920037],
            [0.99999946, 0.99996406, 0.99742985, 0.99122226, 0.99100465],
            [0.9999994, 0.99996006, 0.99714446, 0.99024755, 0.99000573],
        ];

        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).square().mean().backward();
            opt.update(&mut t, gradients).expect("");
            assert_close(&t.as_array(), e);
        }
    }

    #[test]
    fn test_custom_adam_one_params() {
        let dev = build_test_device!();
        let mut opt = Adam::new(AdamConfig {
            lr: 1e-3,
            betas: [0.5, 0.25],
            eps: 1e-8,
            weight_decay: None,
        });
        let mut t: Tensor1D<5, _> = dev.ones();
        let rate = dev.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1e-0]);
        let expected = [
            [0.9997143, 0.9990244, 0.99900025, 0.999, 0.999],
            [0.99942863, 0.99804866, 0.9980004, 0.9979999, 0.9979999],
            [0.999143, 0.9970728, 0.99700034, 0.9969996, 0.9969996],
            [0.99885744, 0.99609685, 0.9960002, 0.9959992, 0.9959992],
            [0.9985719, 0.9951208, 0.9949999, 0.9949987, 0.9949987],
            [0.99828637, 0.9941448, 0.99399954, 0.9939981, 0.9939981],
            [0.9980009, 0.9931687, 0.9929992, 0.9929975, 0.99299747],
            [0.99771553, 0.9921926, 0.9919988, 0.9919969, 0.9919968],
            [0.9974302, 0.9912166, 0.9909984, 0.99099624, 0.9909962],
            [0.99714494, 0.9902406, 0.989998, 0.9899956, 0.98999554],
        ];

        for e in expected.iter() {
            let gradients = (t.trace() * rate.clone()).square().mean().backward();
            opt.update(&mut t, gradients).expect("");
            assert_eq!(&t.as_array(), e);
        }
    }

    #[test]
    fn test_adam_l2_decay() {
        let dev = build_test_device!();
        let mut opt = Adam::new(AdamConfig {
            betas: [0.5, 0.25],
            weight_decay: Some(WeightDecay::L2(1.0)),
            ..Default::default()
        });
        let mut t: Tensor1D<5, _> = dev.tensor([-0.5, -0.25, 0.1, 0.6, 1.0]);
        #[rustfmt::skip]
        let expected = [
            [-0.499, -0.249, 0.099, 0.59900004, 0.999],
            [-0.49799952, -0.24797276, 0.09799955, 0.5979998, 0.9979998],
            [-0.49699846, -0.24689871, 0.09699859, 0.5969993, 0.99699926],
            [-0.49599692,-0.24575013,0.095997185,0.5959985,0.99599856],
            [-0.49499503,-0.24448763,0.094995454,0.5949976,0.9949977],
            [-0.4939929, -0.24382699, 0.09399351, 0.59399647, 0.9939967],
            [-0.49299058, -0.24413459, 0.09299142, 0.5929953, 0.9929956],
            [-0.49198818, -0.24478404, 0.09198925, 0.59199405, 0.9919945],
            [-0.49098572, -0.24561276, 0.09098703, 0.5909928, 0.9909934],
            [-0.48998323, -0.24548599, 0.08998477, 0.58999157, 0.9899922],
        ];

        for e in expected.iter() {
            let gradients = t.trace().exp().square().mean().backward();
            opt.update(&mut t, gradients).expect("");
            assert_eq!(&t.as_array(), e);
        }
    }

    #[test]
    fn test_adam_decoupled_decay() {
        let dev = build_test_device!();
        let mut opt = Adam::new(AdamConfig {
            betas: [0.5, 0.25],
            weight_decay: Some(WeightDecay::Decoupled(1.0)),
            ..Default::default()
        });
        let mut t: Tensor1D<5, _> = dev.tensor([-0.5, -0.25, 0.1, 0.6, 1.0]);
        #[rustfmt::skip]
        let expected = [
            [-0.5005, -0.25075,  0.098900005,  0.5984,  0.998],
            [-0.5009996, -0.25149944,  0.09780081,  0.59680116,  0.9960015],
            [-0.50149894, -0.25224838,  0.09670238,  0.59520346,  0.9940043],
            [-0.5019978, -0.25299674,  0.09560476,  0.59360695,  0.9920086],
            [-0.50249636, -0.2537445,  0.09450804,  0.5920117,  0.99001455],
            [-0.5029944, -0.25449163,  0.09341227,  0.59041786,  0.98802227],
            [-0.50349206, -0.25523806,  0.092317514,  0.58882546,  0.9860318],
            [-0.5039892, -0.25598377,  0.0912238,  0.5872346,  0.9840432],
            [-0.5044859, -0.25672877,  0.09013115,  0.5856453,  0.98205656],
            [-0.50498205, -0.25747302,  0.08903958,  0.58405757,  0.9800719],
        ];

        for e in expected.iter() {
            let gradients = t.trace().exp().square().mean().backward();
            opt.update(&mut t, gradients).expect("");
            assert_eq!(&t.as_array(), e);
        }
    }

    // #[test]
    // fn test_adam_changes_all_params() {
    //     let dev = build_test_device!();
    //     type Model = (Linear<5, 16>, ReLU, Linear<16, 16>, ReLU, Linear<16, 10>);
    //     let mut rng = StdRng::seed_from_u64(0);
    //     let mut model: Model = Default::default();
    //     model.reset_params(&mut rng);
    //     let model_0 = model.clone();

    //     let x: Tensor2D<16, 5> = Tensor2D::rand(&mut rng);
    //     let y: Tensor2D<16, 10> = Tensor2D::rand(&mut rng);
    //     let mut opt: Adam<Model> = Adam::new(AdamConfig {
    //         lr: 1e-3,
    //         betas: [0.9, 0.999],
    //         eps: 1e-8,
    //         weight_decay: None,
    //     });

    //     let py = model.forward(x.trace());
    //     let loss = (py - y).square().mean();
    //     let gradients = loss.backward();
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
    // fn test_adam_unused_params() {
    //     let dev = build_test_device!();
    //     type Model = (Linear<5, 16>, Linear<16, 10>);
    //     let mut model: Model = Default::default();
    //     let mut opt: Adam<Model> = Default::default();
    //     let y = model.1.forward(Tensor2D::<8, 16>::zeros().trace());
    //     let g = y.mean().backward();
    //     opt.update(&mut model, g).expect_err("");
    // }
}
