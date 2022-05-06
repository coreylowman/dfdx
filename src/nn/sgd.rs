use crate::prelude::*;

#[derive(Debug)]
pub struct Sgd {
    lr: f32,
    gradients: Gradients,
}

impl Default for Sgd {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            gradients: Default::default(),
        }
    }
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            gradients: Default::default(),
        }
    }
}

impl GradientProvider for Sgd {
    fn gradient<T: HasUniqueId + IsNdArray>(&mut self, t: &T) -> Option<T::ArrayType> {
        self.gradients
            .remove_gradient(t)
            .map(|g| g.map_elems(|v| v * self.lr))
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

    #[test]
    fn test_sgd_one_params() {
        let mut t: Tensor1D<5> = Tensor1D::ones();
        let mut sgd = Sgd::new(1e-2);
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
        let a: Tensor1D<5> = Tensor1D::new([1.0; 5]);
        let b: Tensor1D<5> = Tensor1D::new([0.5; 5]);
        let mut m = (a, b);
        let mut sgd = Sgd::new(1e-1);

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
        let mut pred: Tensor1D<5> = Tensor1D::zeros();
        let targ: Tensor1D<5> = Tensor1D::ones();
        let mut sgd = Sgd::new(1.0);
        for _ in 0..5 {
            let loss = (pred.trace() - &targ).abs().mean();
            let gradients = loss.backward();
            sgd.update(&mut pred, gradients);
        }
        assert_eq!(pred.data(), &[1.0; 5]);
        assert_eq!(targ.data(), &[1.0; 5]);
    }
}
