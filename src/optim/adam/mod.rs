mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use std::{marker::PhantomData, sync::Arc};

use crate::{
    nn::tensor_collection::*,
    prelude::{Device, Tensor},
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, Gradients},
};

use super::{Optimizer, OptimizerUpdateError, UnusedTensors, WeightDecay};

/// Configuration of hyperparameters for [Adam].
///
/// Changing all default parameters:
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
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

impl<E: Dtype> Default for AdamConfig<E> {
    fn default() -> Self {
        Self {
            lr: E::from_f32(1e-3).unwrap(),
            betas: [E::from_f32(0.9).unwrap(), E::from_f32(0.999).unwrap()],
            eps: E::from_f32(1e-8).unwrap(),
            weight_decay: None,
        }
    }
}

/// An implementation of the Adam optimizer from
/// [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
///
/// # Example Usage
/// ```rust
/// # use dfdx::{prelude::*, optim::*};
/// # type Model = Tensor<Rank0, f32, Cpu>;
/// # let dev: Cpu = Default::default();
/// # let model: Model = dev.zeros();
/// let mut opt: Adam<Model, f32, Cpu> = Adam::new(&model, AdamConfig {
///     lr: 1e-2,
///     betas: [0.5, 0.25],
///     eps: 1e-6,
///     weight_decay: Some(WeightDecay::Decoupled(1e-2)),
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug)]
pub struct Adam<M, E: Dtype, D: DeviceStorage> {
    /// Hyperparameter configuration
    pub cfg: AdamConfig<E>,

    t: i32,
    moment1: Gradients<E, D>,
    moment2: Gradients<E, D>,

    marker: PhantomData<*const M>,
}

impl<M, E: Dtype, D: DeviceStorage> Adam<M, E, D> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(_model: &M, cfg: AdamConfig<E>) -> Self {
        Self {
            cfg,
            t: 0,
            moment1: Gradients::leaky(),
            moment2: Gradients::leaky(),
            marker: PhantomData,
        }
    }
}

pub trait AdamKernel<E: Dtype>: DeviceStorage {
    fn update(
        &self,
        t: i32,
        cfg: &AdamConfig<E>,
        param: &mut Self::Vec<E>,
        moment1: &mut Self::Vec<E>,
        moment2: &mut Self::Vec<E>,
        grad: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

impl<M, D: Device<E>, E: Dtype> TensorVisitor<E, D>
    for (&mut Adam<M, E, D>, &Gradients<E, D>, UnusedTensors)
{
    type Viewer = ViewTensorMut;
    type Err = D::Err;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        p: &mut crate::prelude::Tensor<S, E, D>,
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        if !opts.do_gradient_update {
            return Ok(None);
        }
        let g = self.1.get_ref_checked(p);
        match g {
            None => self.2.add(p),
            Some(g) => {
                let m_t = self.0.moment1.get_or_alloc_mut(p)?;
                let v_t = self.0.moment2.get_or_alloc_mut(p)?;
                AdamKernel::update(
                    &p.device,
                    self.0.t,
                    &self.0.cfg,
                    Arc::make_mut(&mut p.data),
                    m_t,
                    v_t,
                    g,
                )?;
            }
        }
        Ok(None)
    }
}

impl<M: TensorCollection<E, D>, D: Device<E>, E: Dtype> Optimizer<M, D, E> for Adam<M, E, D> {
    fn update(
        &mut self,
        module: &mut M,
        gradients: &Gradients<E, D>,
    ) -> Result<(), OptimizerUpdateError<D>> {
        self.t = self.t.checked_add(1).unwrap();
        let mut op = (self, gradients, Default::default());
        let result = M::iter_tensors(&mut RecursiveWalker {
            m: module,
            f: &mut op,
        });
        match result {
            Ok(_) => op.2.into(),
            Err(e) => Err(OptimizerUpdateError::DeviceError(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::*, tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_default_adam_params() {
        let dev: TestDevice = Default::default();
        let mut t: Tensor<Rank1<5>, TestDtype, _> = dev.ones();
        let mut opt = Adam::new(&t, Default::default());
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
            let gradients = (t.leaky_trace() * rate.clone()).square().mean().backward();
            opt.update(&mut t, &gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_custom_adam_one_params() {
        let dev: TestDevice = Default::default();
        let mut t: Tensor<Rank1<5>, TestDtype, _> = dev.ones();
        let mut opt = Adam::new(
            &t,
            AdamConfig {
                lr: 1e-3,
                betas: [0.5, 0.25],
                eps: 1e-8,
                weight_decay: None,
            },
        );
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
            let gradients = (t.leaky_trace() * rate.clone()).square().mean().backward();
            opt.update(&mut t, &gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_adam_l2_decay() {
        let dev: TestDevice = Default::default();
        let mut t: Tensor<Rank1<5>, TestDtype, _> = dev.tensor([-0.5, -0.25, 0.1, 0.6, 1.0]);
        let mut opt = Adam::new(
            &t,
            AdamConfig {
                betas: [0.5, 0.25],
                weight_decay: Some(WeightDecay::L2(1.0)),
                ..Default::default()
            },
        );
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
            let gradients = t.leaky_trace().exp().square().mean().backward();
            opt.update(&mut t, &gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_adam_decoupled_decay() {
        let dev: TestDevice = Default::default();
        let mut t: Tensor<Rank1<5>, TestDtype, _> = dev.tensor([-0.5, -0.25, 0.1, 0.6, 1.0]);
        let mut opt = Adam::new(
            &t,
            AdamConfig {
                betas: [0.5, 0.25],
                weight_decay: Some(WeightDecay::Decoupled(1.0)),
                ..Default::default()
            },
        );
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
            let gradients = t.leaky_trace().exp().square().mean().backward();
            opt.update(&mut t, &gradients).expect("");
            assert_close(&t.array(), e);
        }
    }

    #[test]
    fn test_unused_tensors() {
        let dev: TestDevice = Default::default();
        let mut t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let mut opt = Adam::new(&t, Default::default());
        opt.update(&mut t, &Gradients::leaky()).expect_err("");
    }
}
