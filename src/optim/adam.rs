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
    moments: [Gradients; 2],
    next_moments: [Gradients; 2],
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
            moments: Default::default(),
            next_moments: Default::default(),
        }
    }
}

impl GradientProvider for Adam {
    fn gradient<P>(&mut self, p: &P) -> Option<Box<P::Array>>
    where
        P: HasUniqueId + HasArrayType<Dtype = f32> + HasDevice,
    {
        self.gradients.remove(p).map(|mut g_t| {
            let mut m_t = self.moments[0].remove(p).unwrap_or_else(P::Device::zeros);
            let mut v_t = self.moments[1].remove(p).unwrap_or_else(P::Device::zeros);
            P::Device::zip_map_assign(m_t.as_mut(), g_t.as_ref(), &mut |m, g| {
                *m = *m * self.betas[0] + g * (1.0 - self.betas[0]);
            });
            P::Device::zip_map_assign(v_t.as_mut(), g_t.as_ref(), &mut |v, g| {
                *v = *v * self.betas[1] + g.powi(2) * (1.0 - self.betas[1]);
            });
            P::Device::zip_map_into(m_t.as_ref(), v_t.as_ref(), g_t.as_mut(), |m, v| {
                let m = m * (1.0 - self.betas[0].powi(self.t)).recip();
                let v = v * (1.0 - self.betas[1].powi(self.t)).recip();
                self.lr * m / (v.sqrt() + self.eps)
            });
            self.next_moments[0].insert(p, m_t);
            self.next_moments[1].insert(p, v_t);
            g_t
        })
    }
}

impl Optimizer for Adam {
    fn update<M: CanUpdateWithGradients>(&mut self, module: &mut M, gradients: Gradients) {
        self.t += 1;
        self.gradients = gradients;
        module.update(self);
        self.moments = std::mem::take(&mut self.next_moments);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::*, SeedableRng};
    use rand_distr::Normal;

    #[test]
    fn test_default_adam_params() {
        let mut opt = Adam::default();
        let mut t: Tensor1D<5> = Tensor1D::ones();
        let rate = Tensor1D::new([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]);
        let expected = [
            [0.99999994, 0.99999601, 0.99971431, 0.99902439, 0.99900025],
            [0.99999988, 0.99999201, 0.99942863, 0.99804884, 0.99800050],
            [0.99999982, 0.99998802, 0.99914300, 0.99707329, 0.99700081],
            [0.99999976, 0.99998403, 0.99885738, 0.99609786, 0.99600118],
            [0.99999970, 0.99998003, 0.99857181, 0.99512249, 0.9950017],
            [0.99999964, 0.99997604, 0.99828625, 0.99414724, 0.9940022],
            [0.99999958, 0.99997205, 0.99800074, 0.99317211, 0.9930029],
            [0.99999952, 0.99996805, 0.99771529, 0.99219710, 0.9920037],
            [0.99999946, 0.99996406, 0.99742985, 0.99122226, 0.99100465],
            [0.99999940, 0.99996006, 0.99714446, 0.99024755, 0.99000573],
        ];

        for e in expected.iter() {
            let gradients = (&rate * t.trace()).square().mean().backward();
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
            [0.99971431, 0.99902439, 0.99900025, 0.99900001, 0.99900001],
            [0.99942863, 0.99804866, 0.99800038, 0.99799991, 0.99799991],
            [0.99914300, 0.99707282, 0.99700034, 0.99699962, 0.99699962],
            [0.99885744, 0.99609685, 0.99600017, 0.99599922, 0.99599922],
            [0.99857187, 0.99512082, 0.99499989, 0.99499869, 0.99499869],
            [0.99828637, 0.99414480, 0.99399954, 0.99399811, 0.99399811],
            [0.99800092, 0.99316871, 0.99299920, 0.99299753, 0.99299747],
            [0.99771553, 0.99219263, 0.99199879, 0.99199688, 0.99199682],
            [0.99743021, 0.99121660, 0.99099839, 0.99099624, 0.99099618],
            [0.99714494, 0.99024057, 0.98999798, 0.98999560, 0.98999554],
        ];

        for e in expected.iter() {
            let gradients = (&rate * t.trace()).square().mean().backward();
            opt.update(&mut t, gradients);
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_adam_changes_all_params() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: (Linear<5, 16>, ReLU, Linear<16, 16>, ReLU, Linear<16, 10>) =
            Default::default();
        model.randomize(&mut rng, &Normal::new(0.0, 0.1).unwrap());
        let model_0 = model.clone();

        let x: Tensor2D<16, 5> = Tensor2D::rand(&mut rng);
        let y: Tensor2D<16, 10> = Tensor2D::rand(&mut rng);
        let mut opt: Adam = Adam::new(1e-3, [0.9, 0.999], 1e-8);

        let py = model.forward(x.trace());
        let loss = (&y - py).square().mean();
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
