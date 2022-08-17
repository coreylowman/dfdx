use crate::prelude::*;
use std::marker::PhantomData;

/// Implementation of Stochastic Gradient Descent. Based on [pytorch's implementation](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
///
/// Nesterov Momentum is implemented as described in
/// [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html).
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
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug)]
pub struct Sgd<M> {
    /// Hyperparameter configuration
    pub cfg: SgdConfig,

    velocity: Gradients,
    gradients: Gradients,

    marker: PhantomData<*const M>,
}

/// Configuration of hyperparameters for [Sgd].
///
/// Using different learning rate:
/// ```rust
/// # use dfdx::prelude::*;
/// SgdConfig {
///     lr: 1e-1,
///     momentum: None
/// };
/// ```
///
/// Using classic momentum:
/// ```rust
/// # use dfdx::prelude::*;
/// SgdConfig {
///     lr: 1e-2,
///     momentum: Some(Momentum::Classic(0.5))
/// };
/// ```
///
/// Using nesterov momentum:
/// ```rust
/// # use dfdx::prelude::*;
/// SgdConfig {
///     lr: 1e-3,
///     momentum: Some(Momentum::Nesterov(0.25))
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SgdConfig {
    /// Learning rate. Defaults to `1e-2`
    pub lr: f32,

    /// Optional momentum. Defaults to `None`.
    pub momentum: Option<Momentum>,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            momentum: None,
        }
    }
}

/// Momentum used for [Sgd]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Momentum {
    /// Momentum that is applied to the velocity of a parameter directly.
    Classic(f32),

    /// Momentum that is applied to both velocity and gradients. See [Sgd] nesterov paper for more.
    Nesterov(f32),
}

impl<M> Default for Sgd<M> {
    /// See [SgdConfig]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<M> Sgd<M> {
    /// Constructs using hyperparameters from `cfg`
    pub fn new(cfg: SgdConfig) -> Self {
        Self {
            cfg,
            velocity: Default::default(),
            gradients: Default::default(),
            marker: PhantomData,
        }
    }
}

impl<M> GradientProvider for Sgd<M> {
    fn gradient<P>(&mut self, p: &P) -> Option<Box<P::Array>>
    where
        P: HasUniqueId + HasArrayType<Dtype = f32> + HasDevice,
    {
        let mut g_t = self.gradients.remove(p)?;
        match self.cfg.momentum {
            Some(Momentum::Classic(u)) => {
                let v_t = self.velocity.mut_gradient(p);
                P::Device::foreach_mm(g_t.as_mut(), v_t, &mut |g, v| {
                    *v = *g + u * *v;
                    *g = *v * self.cfg.lr;
                });
            }
            Some(Momentum::Nesterov(u)) => {
                let v_t = self.velocity.mut_gradient(p);
                P::Device::foreach_mm(g_t.as_mut(), v_t, &mut |g, v| {
                    *v = *g + u * *v;
                    *g = (*g + u * *v) * self.cfg.lr;
                });
            }
            None => P::Device::foreach_m(g_t.as_mut(), &mut |g| *g *= self.cfg.lr),
        }
        Some(g_t)
    }
}

impl<M: CanUpdateWithGradients> Optimizer<M> for Sgd<M> {
    fn update(&mut self, module: &mut M, gradients: Gradients) -> Result<(), UnusedParamsError> {
        let mut missing = Default::default();
        self.gradients = gradients;
        module.update(self, &mut missing);
        missing.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::StdRng, SeedableRng};

    #[test]
    fn test_perfect_sgd() {
        let mut sgd = Sgd::new(SgdConfig {
            lr: 1.0,
            momentum: None,
        });

        let mut pred: Tensor1D<5> = Tensor1D::zeros();
        let targ: Tensor1D<5> = Tensor1D::ones();
        for _ in 0..5 {
            let loss = (pred.trace() - &targ).abs().mean();
            let gradients = loss.backward();
            sgd.update(&mut pred, gradients).expect("");
        }
        assert_eq!(pred.data(), &[1.0; 5]);
        assert_eq!(targ.data(), &[1.0; 5]);
    }

    #[test]
    fn test_sgd_no_momentum() {
        let mut sgd = Sgd::new(Default::default());

        let mut t: Tensor1D<5> = Tensor1D::ones();
        let rate = Tensor1D::new([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9998, 0.998, 0.996, 0.98, 0.8],
            [0.99960005, 0.99600005, 0.992, 0.96000004, 0.6],
            [0.9994001, 0.9940001, 0.988, 0.94000006, 0.40000004],
            [0.9992001, 0.9920001, 0.98399997, 0.9200001, 0.20000005],
            [0.99900013, 0.9900001, 0.97999996, 0.9000001, 5.9604645e-8],
        ];

        for e in expected.iter() {
            let gradients = (t.trace() * &rate).mean().backward();
            sgd.update(&mut t, gradients).expect("");
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_sgd_classic_momentum() {
        let mut sgd = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: Some(Momentum::Classic(0.5)),
        });

        let mut t: Tensor1D<5> = Tensor1D::ones();
        let rate = Tensor1D::new([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9998, 0.998, 0.996, 0.98, 0.8],
            [0.99950004, 0.995, 0.99, 0.95000005, 0.5],
            [0.99915004, 0.9915, 0.983, 0.915, 0.15],
            [0.99877506, 0.98775, 0.9755, 0.8775, -0.225],
            [0.9983876, 0.983875, 0.96775, 0.83875, -0.61249995],
        ];

        for e in expected.iter() {
            let gradients = (t.trace() * &rate).mean().backward();
            sgd.update(&mut t, gradients).expect("");
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_sgd_nesterov_momentum() {
        let mut sgd = Sgd::new(SgdConfig {
            lr: 1e-2,
            momentum: Some(Momentum::Nesterov(0.5)),
        });

        let mut t: Tensor1D<5> = Tensor1D::ones();
        let rate = Tensor1D::new([0.1, 1.0, 2.0, 10.0, 100.0]);
        let expected = [
            [0.9997, 0.997, 0.994, 0.97, 0.70000005],
            [0.99935, 0.9935, 0.987, 0.935, 0.35000005],
            [0.99897504, 0.98974997, 0.9795, 0.8975, -0.024999946],
            [0.99858755, 0.98587495, 0.97175, 0.85875, -0.41249993],
            [0.9981938, 0.98193747, 0.963875, 0.819375, -0.8062499],
        ];

        for e in expected.iter() {
            let gradients = (t.trace() * &rate).mean().backward();
            sgd.update(&mut t, gradients).expect("");
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_sgd_changes_all_params() {
        type Model = (Linear<5, 16>, ReLU, Linear<16, 16>, ReLU, Linear<16, 10>);
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: Model = Default::default();
        model.reset_params(&mut rng);
        let model_0 = model.clone();

        let x: Tensor2D<16, 5> = Tensor2D::rand(&mut rng);
        let y: Tensor2D<16, 10> = Tensor2D::rand(&mut rng);
        let mut opt: Sgd<Model> = Default::default();

        let py = model.forward(x.trace());
        let loss = (py - &y).square().mean();
        let gradients = loss.backward();
        opt.update(&mut model, gradients).expect("");

        let model_1 = model.clone();

        assert!(model_0.0.weight.data() != model_1.0.weight.data());
        assert!(model_0.0.bias.data() != model_1.0.bias.data());
        assert!(model_0.2.weight.data() != model_1.2.weight.data());
        assert!(model_0.2.bias.data() != model_1.2.bias.data());
        assert!(model_0.4.weight.data() != model_1.4.weight.data());
        assert!(model_0.4.bias.data() != model_1.4.bias.data());
    }
}
