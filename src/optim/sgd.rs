use crate::prelude::*;

/// Implementation of Stochastic Gradient Descent. Based on [pytorch's implementation](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
///
/// Nesterov Momentum is implemented as described in
/// [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html).
///
/// Example Usage:
/// ```rust
/// use dfdx::prelude::*;
///
/// let mut t = Tensor0D::ones();
/// let mut opt: Sgd = Default::default();
///
/// let gradients = t.trace().backward();
/// opt.update(&mut t, gradients);
/// ```
///
/// Changing default parmeters:
/// ```rust
/// use dfdx::optim::{Sgd, Momentum};
///
/// let sgd_no_momentum = Sgd::new(1e-1, None);
/// let sgd_classic_momentum = Sgd::new(1e-2, Some(Momentum::Classic(0.5)));
/// let sgd_nesterov_momentum = Sgd::new(1e-3, Some(Momentum::Nesterov(0.25)));
/// ```
#[derive(Debug)]
pub struct Sgd {
    pub lr: f32,
    pub momentum: Option<Momentum>,
    velocity: Gradients,
    gradients: Gradients,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Momentum {
    Classic(f32),
    Nesterov(f32),
}

impl Default for Sgd {
    /// - `self.lr = 1e-2`
    /// - `self.momentum = None`
    fn default() -> Self {
        Self::new(1e-2, None)
    }
}

impl Sgd {
    /// Constructs [Sgd] with specified learning rate and momentum.
    pub fn new(lr: f32, momentum: Option<Momentum>) -> Self {
        Self {
            lr,
            momentum,
            velocity: Default::default(),
            gradients: Default::default(),
        }
    }
}

impl GradientProvider for Sgd {
    fn gradient<P>(&mut self, p: &P) -> Box<P::Array>
    where
        P: HasUniqueId + HasArrayType<Dtype = f32> + HasDevice,
    {
        let mut g_t = self.gradients.remove(p);
        match self.momentum {
            Some(Momentum::Classic(u)) => {
                let v_t = self.velocity.mut_gradient(p);
                P::Device::foreach_mm(g_t.as_mut(), v_t, &mut |g, v| {
                    *v = *g + u * *v;
                    *g = *v * self.lr;
                });
            }
            Some(Momentum::Nesterov(u)) => {
                let v_t = self.velocity.mut_gradient(p);
                P::Device::foreach_mm(g_t.as_mut(), v_t, &mut |g, v| {
                    *v = *g + u * *v;
                    *g = (*g + u * *v) * self.lr;
                });
            }
            None => P::Device::foreach_m(g_t.as_mut(), &mut |g| *g *= self.lr),
        }
        g_t
    }
}

impl Optimizer for Sgd {
    fn update<M: CanUpdateWithGradients>(&mut self, module: &mut M, gradients: Gradients) {
        self.gradients = gradients;
        module.update(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::StdRng, SeedableRng};

    #[test]
    fn test_perfect_sgd() {
        let mut sgd = Sgd::new(1.0, None);

        let mut pred: Tensor1D<5> = Tensor1D::zeros();
        let targ: Tensor1D<5> = Tensor1D::ones();
        for _ in 0..5 {
            let loss = (pred.trace() - &targ).abs().mean();
            let gradients = loss.backward();
            sgd.update(&mut pred, gradients);
        }
        assert_eq!(pred.data(), &[1.0; 5]);
        assert_eq!(targ.data(), &[1.0; 5]);
    }

    #[test]
    fn test_sgd_no_momentum() {
        let mut sgd = Sgd::new(1e-2, None);

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
            sgd.update(&mut t, gradients);
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_sgd_classic_momentum() {
        let mut sgd = Sgd::new(1e-2, Some(Momentum::Classic(0.5)));

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
            sgd.update(&mut t, gradients);
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_sgd_nesterov_momentum() {
        let mut sgd = Sgd::new(1e-2, Some(Momentum::Nesterov(0.5)));

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
            sgd.update(&mut t, gradients);
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_sgd_changes_all_params() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: (Linear<5, 16>, ReLU, Linear<16, 16>, ReLU, Linear<16, 10>) =
            Default::default();
        model.reset_params(&mut rng);
        let model_0 = model.clone();

        let x: Tensor2D<16, 5> = Tensor2D::rand(&mut rng);
        let y: Tensor2D<16, 10> = Tensor2D::rand(&mut rng);
        let mut opt: Sgd = Default::default();

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
