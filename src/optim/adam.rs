use crate::prelude::*;

/// An implementation of the Adam optimizer from
/// [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
///
/// Example Usage:
/// ```rust
/// use dfdx::prelude::*;
///
/// let mut t = Tensor0D::ones();
/// let mut opt: Adam = Default::default();
///
/// let gradients = t.trace().backward();
/// opt.update(&mut t, gradients);
/// ```
///
/// Changing default parmeters:
/// ```rust
/// use dfdx::optim::Adam;
///
/// let adam = Adam::new(1e-2, [0.5, 0.25], 1e-6);
/// ```
#[derive(Debug)]
pub struct Adam {
    pub lr: f32,
    pub betas: [f32; 2],
    pub eps: f32,
    t: i32,
    gradients: Gradients,
    moment1: Gradients,
    moment2: Gradients,
}

/// Use the default parameters suggested in the paper of lr=1e-3, beta1=0.9, beta2=0.999, and epsilon=1e-8
impl Default for Adam {
    fn default() -> Self {
        Self::new(1e-3, [0.9, 0.999], 1e-8)
    }
}

impl Adam {
    pub fn new(lr: f32, betas: [f32; 2], epsilon: f32) -> Self {
        Self {
            lr,
            betas,
            eps: epsilon,
            t: 0,
            gradients: Default::default(),
            moment1: Default::default(),
            moment2: Default::default(),
        }
    }
}

impl GradientProvider for Adam {
    fn gradient<P>(&mut self, p: &P) -> Box<P::Array>
    where
        P: HasUniqueId + HasArrayType<Dtype = f32> + HasDevice,
    {
        let mut g_t = self.gradients.remove(p);
        let m_t = self.moment1.mut_gradient(p);
        let v_t = self.moment2.mut_gradient(p);
        P::Device::foreach_mmm(g_t.as_mut(), m_t, v_t, &mut |g, m, v| {
            *m = *m * self.betas[0] + *g * (1.0 - self.betas[0]);
            *v = *v * self.betas[1] + g.powi(2) * (1.0 - self.betas[1]);
            let m_hat = *m * (1.0 - self.betas[0].powi(self.t)).recip();
            let v_hat = *v * (1.0 - self.betas[1].powi(self.t)).recip();
            *g = self.lr * m_hat / (v_hat.sqrt() + self.eps)
        });
        g_t
    }
}

impl Optimizer for Adam {
    fn update<M: CanUpdateWithGradients>(&mut self, module: &mut M, gradients: Gradients) {
        self.t = self.t.checked_add(1).unwrap();
        self.gradients = gradients;
        module.update(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::*, SeedableRng};

    #[test]
    fn test_default_adam_params() {
        let mut opt = Adam::default();
        let mut t: Tensor1D<5> = Tensor1D::ones();
        let rate = Tensor1D::new([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]);
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
            opt.update(&mut t, gradients);
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_custom_adam_one_params() {
        let mut opt = Adam::new(1e-3, [0.5, 0.25], 1e-8);
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
            opt.update(&mut t, gradients);
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_adam_changes_all_params() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: (Linear<5, 16>, ReLU, Linear<16, 16>, ReLU, Linear<16, 10>) =
            Default::default();
        model.reset_params(&mut rng);
        let model_0 = model.clone();

        let x: Tensor2D<16, 5> = Tensor2D::rand(&mut rng);
        let y: Tensor2D<16, 10> = Tensor2D::rand(&mut rng);
        let mut opt: Adam = Adam::new(1e-3, [0.9, 0.999], 1e-8);

        let py = model.forward(x.trace());
        let loss = (py - &y).square().mean();
        let gradients = loss.backward();
        opt.update(&mut model, gradients);

        let model_1 = model.clone();

        assert!(model_0.0.weight.data() != model_1.0.weight.data());
        assert!(model_0.0.bias.data() != model_1.0.bias.data());
        assert!(model_0.2.weight.data() != model_1.2.weight.data());
        assert!(model_0.2.bias.data() != model_1.2.bias.data());
        assert!(model_0.4.weight.data() != model_1.4.weight.data());
        assert!(model_0.4.bias.data() != model_1.4.bias.data());
    }
}
