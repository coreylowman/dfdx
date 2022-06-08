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
    next_velocity: Gradients,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Momentum {
    Classic(f32),
    Nesterov(f32),
}

impl Default for Sgd {
    fn default() -> Self {
        Self::new(1e-2, None)
    }
}

impl Sgd {
    pub fn new(lr: f32, momentum: Option<Momentum>) -> Self {
        Self {
            lr,
            momentum,
            velocity: Default::default(),
            gradients: Default::default(),
            next_velocity: Default::default(),
        }
    }
}

impl GradientProvider for Sgd {
    fn gradient<P>(&mut self, p: &P) -> Option<Box<P::Array>>
    where
        P: HasUniqueId + HasArrayType<Dtype = f32> + HasDevice,
    {
        self.gradients.remove(p).map(|mut g_t| {
            match self.momentum {
                Some(Momentum::Classic(u)) => {
                    let mut v_t = self.velocity.remove(p).unwrap_or_else(P::Device::zeros);
                    P::Device::zip_map_assign(v_t.as_mut(), g_t.as_ref(), &mut |v, g| {
                        *v = g + u * *v
                    });
                    P::Device::zip_map_assign(g_t.as_mut(), v_t.as_ref(), &mut |g, v| {
                        *g = v * self.lr
                    });
                    self.next_velocity.insert(p, v_t);
                }
                Some(Momentum::Nesterov(u)) => {
                    let mut v_t = self.velocity.remove(p).unwrap_or_else(P::Device::zeros);
                    P::Device::zip_map_assign(v_t.as_mut(), g_t.as_ref(), &mut |v, g| {
                        *v = g + u * *v
                    });
                    P::Device::zip_map_assign(g_t.as_mut(), v_t.as_ref(), &mut |g, v| {
                        *g = (*g + u * v) * self.lr
                    });
                    self.next_velocity.insert(p, v_t);
                }
                None => P::Device::map_assign(g_t.as_mut(), &mut |g| *g *= self.lr),
            }
            g_t
        })
    }
}

impl Optimizer for Sgd {
    fn update<M: CanUpdateWithGradients>(&mut self, module: &mut M, gradients: Gradients) {
        self.gradients = gradients;
        module.update(self);
        self.velocity = std::mem::take(&mut self.next_velocity);
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
            let loss = (&targ - pred.trace()).abs().mean();
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
            [0.99980003, 0.99800003, 0.99599999, 0.98000002, 0.80000001],
            [0.99960005, 0.99600005, 0.99199998, 0.96000004, 0.60000002],
            [0.99940008, 0.99400008, 0.98799998, 0.94000006, 0.40000004],
            [0.99920011, 0.99200010, 0.98399997, 0.92000008, 0.20000005],
            [0.99900013, 0.99000013, 0.97999996, 0.90000010, 5.9604645e-8],
        ];

        for e in expected.iter() {
            let gradients = (&rate * t.trace()).mean().backward();
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
            [0.99980003, 0.99800003, 0.99599999, 0.98000002, 0.80000001],
            [0.99950004, 0.99500000, 0.99000001, 0.95000005, 0.50000000],
            [0.99915004, 0.99150002, 0.98299998, 0.91500002, 0.15000001],
            [0.99877506, 0.98774999, 0.97549999, 0.87750000, -0.22499999],
            [0.99838758, 0.98387498, 0.96775001, 0.83875000, -0.61249995],
        ];

        for e in expected.iter() {
            let gradients = (&rate * t.trace()).mean().backward();
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
            [0.99970001, 0.99699998, 0.99400002, 0.97000003, 0.70000005],
            [0.99935001, 0.99349999, 0.98699999, 0.93500000, 0.35000005],
            [0.99897504, 0.98974997, 0.97950000, 0.89749998, -0.024999946],
            [0.99858755, 0.98587495, 0.97175002, 0.85874999, -0.41249993],
            [0.99819380, 0.98193747, 0.96387500, 0.81937498, -0.80624992],
        ];

        for e in expected.iter() {
            let gradients = (&rate * t.trace()).mean().backward();
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
