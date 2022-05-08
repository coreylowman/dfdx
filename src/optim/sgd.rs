use crate::prelude::*;

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
    fn gradient<T: HasUniqueId + IsNdArray>(&mut self, t: &T) -> Option<T::ArrayType> {
        self.gradients
            .remove_gradient(t)
            .map(|g| match self.momentum {
                Some(m) => {
                    let v = self
                        .velocity
                        .remove_gradient(t)
                        .unwrap_or(T::ArrayType::ZEROS);

                    let new_v = match m {
                        Momentum::Classic(μ) => v.scale(μ).add(&g),
                        Momentum::Nesterov(μ) => v.scale(μ).add(&g),
                    };
                    *self.next_velocity.mut_gradient(t) = new_v.clone();

                    match m {
                        Momentum::Classic(_) => new_v.scale(self.lr),
                        Momentum::Nesterov(μ) => g.add(&new_v.scale(μ)).scale(self.lr),
                    }
                }
                None => g.scale(self.lr),
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

    #[test]
    fn test_sgd_one_params() {
        let mut sgd = Sgd::new(1e-2, None);

        let mut t: Tensor1D<5> = Tensor1D::ones();

        let gradients = t.trace().mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.2; 5]);
        sgd.update(&mut t, gradients);
        assert_eq!(t.data(), &[0.998; 5]);

        let gradients = t.trace().mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.2; 5]);
        sgd.update(&mut t, gradients);
        assert_eq!(t.data(), &[0.99600005; 5]);

        let gradients = t.trace().mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &[0.2; 5]);
        sgd.update(&mut t, gradients);
        assert_eq!(t.data(), &[0.9940001; 5]);
    }

    #[test]
    fn test_sgd_two_params() {
        let mut sgd = Sgd::new(1e-1, None);

        let a: Tensor1D<5> = Tensor1D::new([1.0; 5]);
        let b: Tensor1D<5> = Tensor1D::new([0.5; 5]);
        let mut m = (a, b);

        let gradients = (m.0.trace() / &m.1).mean().backward();
        sgd.update(&mut m, gradients);

        assert_eq!(m.0.data(), &[0.95999998; 5]);
        assert_eq!(m.1.data(), &[0.57999998; 5]);

        let gradients = (m.0.trace() / &m.1).mean().backward();
        sgd.update(&mut m, gradients);

        assert_eq!(m.0.data(), &[0.92551720; 5]);
        assert_eq!(m.1.data(), &[0.63707489; 5]);
    }

    #[test]
    fn test_sgd_loop_usage() {
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
    fn test_sgd_classic_momentum() {
        let mut sgd = Sgd::new(1e-2, Some(Momentum::Classic(0.5)));
        let mut t: Tensor1D<5> = Tensor1D::ones();
        let expected_values = [0.998, 0.995, 0.9915];

        for &expected in expected_values.iter() {
            let gradients = t.trace().mean().backward();
            sgd.update(&mut t, gradients);
            assert_eq!(t.data(), &[expected; 5]);
        }
    }

    #[test]
    fn test_sgd_nesterov_momentum() {
        let mut sgd = Sgd::new(1e-2, Some(Momentum::Nesterov(0.5)));
        let mut t: Tensor1D<5> = Tensor1D::ones();
        let expected_values = [0.997, 0.9935, 0.98974997, 0.98587495];

        for &expected in expected_values.iter() {
            let gradients = t.trace().mean().backward();
            sgd.update(&mut t, gradients);
            assert_eq!(t.data(), &[expected; 5]);
        }
    }
}
