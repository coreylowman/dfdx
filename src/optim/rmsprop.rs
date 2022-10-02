use crate::prelude::*;
use std::marker::PhantomData;

/// RMSprop As described in [Hinton, 2012](http://www.cs.toronto.edu/%7Etijmen/csc321/slides/lecture_slides_lec6.pdf).
///
/// This implementation is based off of RMSprop from
/// [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/rmsprop_tf.py)
/// because the pytorch implementation has [some issues](https://github.com/pytorch/pytorch/issues/23796).
///
/// The main difference between the pytorch implementation is that [RMSpropConfig::eps] is added inside of the sqrt()
/// operation.
///
/// The `lr_in_momentum` option is not provided because it didn't seem to make a difference in testing.
///
/// # Example Usage
///
/// Constructing using default:
/// ```rust
/// # use dfdx::prelude::*;
/// # type Model = Tensor0D;
/// let mut opt: RMSprop<Model> = Default::default();
/// ```
///
/// Constructing using new:
/// ```rust
/// # use dfdx::prelude::*;
/// # type Model = Tensor0D;
/// let rmsprop: RMSprop<Model> = RMSprop::new(RMSpropConfig {
///     lr: 1e-3,
///     alpha: 0.5,
///     eps: 1e-8,
///     momentum: Some(0.5),
///     centered: false,
/// });
/// ```
///
/// See module level documentation at [crate::optim] for examples of how to actually use an optimizer.
#[derive(Debug)]
pub struct RMSprop<M> {
    /// Hyperparameter configuration
    pub cfg: RMSpropConfig,

    step: usize,
    momentums: Gradients,
    square_avg: Gradients,
    grad_avg: Gradients,
    gradients: Gradients,

    marker: PhantomData<*const M>,
}

/// Configuration of hyperparameters for [RMSprop].
#[derive(Debug, Clone, Copy)]
pub struct RMSpropConfig {
    /// Learning rate. Defaults to `1e-2`.
    pub lr: f32,

    /// Value for exponential moving average. Defaults to `0.9`.
    pub alpha: f32,

    /// Epsilon for stability. Defaults to `1e-8`.
    pub eps: f32,

    /// Optional momentum. Defaults to `None`.
    pub momentum: Option<f32>,

    /// Whether the avg should be centered by the grad's avg value.
    /// Defaults to `false`.
    pub centered: bool,
}

impl Default for RMSpropConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
            momentum: None,
            centered: false,
        }
    }
}

impl<M> Default for RMSprop<M> {
    /// See [RMSpropConfig]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<M> RMSprop<M> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(cfg: RMSpropConfig) -> Self {
        Self {
            cfg,
            step: 0,
            momentums: Default::default(),
            square_avg: Default::default(),
            grad_avg: Default::default(),
            gradients: Default::default(),
            marker: PhantomData,
        }
    }
}

impl<M> GradientProvider for RMSprop<M> {
    fn gradient<P>(&mut self, p: &P) -> Option<Box<P::Array>>
    where
        P: HasUniqueId + HasArrayType<Dtype = f32> + HasDevice,
    {
        let mut g_t = self.gradients.remove(p)?;

        let square_avg = self.square_avg.mut_gradient(p);
        if self.step == 0 {
            P::Device::fill(square_avg, &mut |v| *v = 1.0);
        }

        P::Device::foreach_mr(square_avg, g_t.as_ref(), &mut |sa, g| {
            // sa = a * sa + (1 - a) * g^2
            *sa += (1.0 - self.cfg.alpha) * (g * g - *sa)
        });

        // **NOTE: difference in implementation**
        // instead of allocating a new array for `avg` and then later dividing g_t by avg,
        // here we directly mutate g_t
        if self.cfg.centered {
            let grad_avg = self.grad_avg.mut_gradient(p);
            P::Device::foreach_mmm(g_t.as_mut(), square_avg, grad_avg, &mut |g, sa, ga| {
                // ga = a * ga + (1 - a) * g
                *ga += (1.0 - self.cfg.alpha) * (*g - *ga);
                // NOTE: self.eps in sqrt
                let avg = (*sa - ga.powi(2) + self.cfg.eps).sqrt();
                *g /= avg;
            });
        } else {
            P::Device::foreach_mr(g_t.as_mut(), square_avg, &mut |g, sa| {
                // NOTE: self.eps in sqrt
                let avg = (sa + self.cfg.eps).sqrt();
                *g /= avg;
            });
        };

        match self.cfg.momentum {
            Some(u) => {
                let m_t = self.momentums.mut_gradient(p);
                P::Device::foreach_mm(m_t, g_t.as_mut(), &mut |m, g| {
                    *m = *m * u + *g;
                    *g = *m * self.cfg.lr;
                });
            }
            None => P::Device::foreach_m(g_t.as_mut(), &mut |g| *g *= self.cfg.lr),
        }
        Some(g_t)
    }
}

impl<M: CanUpdateWithGradients> Optimizer<M> for RMSprop<M> {
    fn update(&mut self, module: &mut M, gradients: Gradients) -> Result<(), UnusedParamsError> {
        self.gradients = gradients;
        let mut unused_tensors = Default::default();
        module.update(self, &mut unused_tensors);
        self.step += 1;
        unused_tensors.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_matches_expected(cfg: RMSpropConfig, expected: [[f32; 5]; 5]) {
        let rate = Tensor1D::new([0.1, 1.0, 2.0, 10.0, 100.0]);
        let mut t: Tensor1D<5> = Tensor1D::ones();
        let mut opt = RMSprop::new(cfg);
        for e in expected.iter() {
            let gradients = backward((t.trace() * &rate).square().sum());
            opt.update(&mut t, gradients).expect("");
            assert_eq!(t.data(), e);
        }
    }

    #[test]
    fn test_rmsprop_default() {
        const CFG: RMSpropConfig = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
            momentum: None,
            centered: false,
        };
        const EXPECTED: [[f32; 5]; 5] = [
            [0.9997892, 0.98245883, 0.9703907, 0.9683808, 0.96837723],
            [0.99956703, 0.96670717, 0.9485176, 0.9457928, 0.945788],
            [0.9993329, 0.9521923, 0.9301649, 0.9270585, 0.9270531],
            [0.9990862, 0.9385879, 0.9138966, 0.9105493, 0.91054344],
            [0.9988262, 0.9256831, 0.8990271, 0.8955128, 0.8955067],
        ];
        test_matches_expected(CFG, EXPECTED);
    }

    #[test]
    fn test_rmsprop_momentum() {
        const CFG: RMSpropConfig = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
            momentum: Some(0.9),
            centered: false,
        };
        const EXPECTED: [[f32; 5]; 5] = [
            [0.9997892, 0.98245883, 0.9703907, 0.9683808, 0.96837723],
            [0.9993773, 0.9509201, 0.9218692, 0.9173355, 0.9173275],
            [0.9987725, 0.9082085, 0.86019397, 0.8530321, 0.8530196],
            [0.9979816, 0.8566451, 0.78923434, 0.7795164, 0.7794995],
            [0.9970101, 0.798177, 0.71185935, 0.69974965, 0.6997286],
        ];
        test_matches_expected(CFG, EXPECTED);
    }

    #[test]
    fn test_rmsprop_diff_alpha() {
        const CFG: RMSpropConfig = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.5,
            eps: 1e-8,
            momentum: None,
            centered: false,
        };
        const EXPECTED: [[f32; 5]; 5] = [
            [0.99971724, 0.9873509, 0.9859671, 0.985858, 0.98585784],
            [0.9993176, 0.9763115, 0.97450525, 0.97436625, 0.97436607],
            [0.9987531, 0.96588355, 0.9639029, 0.9637519, 0.96375173],
            [0.99795645, 0.95572895, 0.95366806, 0.95351166, 0.9535115],
            [0.99683434, 0.9457051, 0.9436056, 0.9434466, 0.9434464],
        ];
        test_matches_expected(CFG, EXPECTED);
    }

    #[test]
    fn test_rmsprop_diff_eps() {
        const CFG: RMSpropConfig = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-2,
            momentum: None,
            centered: false,
        };
        const EXPECTED: [[f32; 5]; 5] = [
            [0.9997904, 0.98252594, 0.97041094, 0.9683808, 0.96837723],
            [0.99956954, 0.9668238, 0.9485463, 0.94579285, 0.945788],
            [0.999337, 0.95234853, 0.93019867, 0.9270586, 0.9270531],
            [0.99909216, 0.9387773, 0.9139341, 0.91054934, 0.91054344],
            [0.9988343, 0.9259014, 0.89906746, 0.8955129, 0.8955067],
        ];
        test_matches_expected(CFG, EXPECTED);
    }

    #[test]
    fn test_rmsprop_centered() {
        const CFG: RMSpropConfig = RMSpropConfig {
            lr: 1e-2,
            alpha: 0.9,
            eps: 1e-8,
            momentum: None,
            centered: true,
        };
        const EXPECTED: [[f32; 5]; 5] = [
            [0.9997892, 0.98218256, 0.96900064, 0.9666708, 0.9666667],
            [0.99956703, 0.965664, 0.9448974, 0.941596, 0.9415902],
            [0.9993329, 0.9498438, 0.9236177, 0.91970736, 0.91970056],
            [0.9990862, 0.93438274, 0.90377975, 0.89941716, 0.8994096],
            [0.9988262, 0.9190646, 0.8847198, 0.87998855, 0.8799804],
        ];
        test_matches_expected(CFG, EXPECTED);
    }

    #[test]
    fn test_rmsprop_unused_params() {
        type Model = (Linear<5, 16>, Linear<16, 10>);
        let mut model: Model = Default::default();
        let mut opt: RMSprop<Model> = Default::default();
        let y = model.1.forward(Tensor2D::<8, 16>::zeros().trace());
        let g = backward(y.mean());
        opt.update(&mut model, g).expect_err("");
    }
}
