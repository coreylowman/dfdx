use crate::prelude::*;
use std::marker::PhantomData;

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
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug)]
pub struct Adam<M> {
    /// Hyperparameter configuration
    pub cfg: AdamConfig,

    t: i32,
    gradients: Gradients,
    moment1: Gradients,
    moment2: Gradients,

    marker: PhantomData<*const M>,
}

/// Configuration of hyperparameters for [Adam].
///
/// Changing all default parameters:
/// ```rust
/// # use dfdx::prelude::*;
/// AdamConfig {
///     lr: 1e-2,
///     betas: [0.1, 0.2],
///     eps: 1e-6,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    /// Learning rate. Defaults to `1e-3`.
    pub lr: f32,

    /// Betas from Adam paper. Defaults to `[0.9, 0.999]`.
    pub betas: [f32; 2],

    /// Epsilon for numerical stability. Defaults to `1e-8`.
    pub eps: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
        }
    }
}

impl<M> Default for Adam<M> {
    /// See [AdamConfig]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<M> Adam<M> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(cfg: AdamConfig) -> Self {
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

impl<M> GradientProvider for Adam<M> {
    fn gradient<P>(&mut self, p: &P) -> Option<Box<P::Array>>
    where
        P: HasUniqueId + HasArrayType<Dtype = f32> + HasDevice,
    {
        let mut g_t = self.gradients.remove(p)?;
        let m_t = self.moment1.mut_gradient(p);
        let v_t = self.moment2.mut_gradient(p);
        P::Device::foreach_mmm(g_t.as_mut(), m_t, v_t, &mut |g, m, v| {
            *m = *m * self.cfg.betas[0] + *g * (1.0 - self.cfg.betas[0]);
            *v = *v * self.cfg.betas[1] + g.powi(2) * (1.0 - self.cfg.betas[1]);
            let m_hat = *m * (1.0 - self.cfg.betas[0].powi(self.t)).recip();
            let v_hat = *v * (1.0 - self.cfg.betas[1].powi(self.t)).recip();
            *g = self.cfg.lr * m_hat / (v_hat.sqrt() + self.cfg.eps)
        });
        Some(g_t)
    }
}

impl<M: CanUpdateWithGradients> Optimizer<M> for Adam<M> {
    fn update(&mut self, module: &mut M, gradients: Gradients) -> Result<(), UnusedParamsError> {
        self.t = self.t.checked_add(1).unwrap();
        self.gradients = gradients;
        let mut unused_tensors = Default::default();
        module.update(self, &mut unused_tensors);
        unused_tensors.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{prelude::*, SeedableRng};

    #[test]
    fn test_default_adam_params() {
        let mut opt = Adam::default();
        let mut t: Tensor1D<5> = Tensor1D::ones();
        let rate = tensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]);
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
            let gradients = (t.trace() * &rate).square().mean().backward();
            opt.update(&mut t, gradients).expect("");
            assert_close(t.data(), e);
        }
    }

    #[test]
    fn test_custom_adam_one_params() {
        let mut opt: Adam<Tensor1D<5>> = Adam::new(AdamConfig {
            lr: 1e-3,
            betas: [0.5, 0.25],
            eps: 1e-8,
        });
        let mut t: Tensor1D<5> = Tensor1D::ones();
        let rate = Tensor1D::new([1e-4, 1e-3, 1e-2, 1e-1, 1e-0]);
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
            let gradients = (t.trace() * &rate).square().mean().backward();
            opt.update(&mut t, gradients).expect("");
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_adam_changes_all_params() {
        type Model = (Linear<5, 16>, ReLU, Linear<16, 16>, ReLU, Linear<16, 10>);
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: Model = Default::default();
        model.reset_params(&mut rng);
        let model_0 = model.clone();

        let x: Tensor2D<16, 5> = Tensor2D::rand(&mut rng);
        let y: Tensor2D<16, 10> = Tensor2D::rand(&mut rng);
        let mut opt: Adam<Model> = Adam::new(AdamConfig {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
        });

        let py = model.forward(x.trace());
        let loss = (py - &y).square().mean::<_, AllAxes>();
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

    #[test]
    fn test_adam_unused_params() {
        type Model = (Linear<5, 16>, Linear<16, 10>);
        let mut model: Model = Default::default();
        let mut opt: Adam<Model> = Default::default();
        let y = model.1.forward(Tensor2D::<8, 16>::zeros().trace());
        let g = y.mean::<_, AllAxes>().backward();
        opt.update(&mut model, g).expect_err("");
    }
}
