use super::*;
use crate::prelude::CanUpdateWithGradients;

/// Repeats `T` `N` times. This requires that `T`'s input is the same as it's output.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// type Model = Repeated<(Linear<10, 10>, ReLU), 5>;
/// let model: Model = Default::default();
/// let out: Tensor1D<10> = model.forward(Tensor1D::zeros());
/// ```
#[derive(Debug, Clone)]
pub struct Repeated<T, const N: usize> {
    pub modules: [T; N],
}

impl<T: Default, const N: usize> Default for Repeated<T, N>
where
    [T; N]: Default,
{
    fn default() -> Self {
        Self {
            modules: Default::default(),
        }
    }
}

impl<T: ResetParams, const N: usize> ResetParams for Repeated<T, N> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        for i in 0..N {
            self.modules[i].reset_params(rng);
        }
    }
}

impl<T: CanUpdateWithGradients, const N: usize> CanUpdateWithGradients for Repeated<T, N> {
    fn update<G: crate::prelude::GradientProvider>(&mut self, grads: &mut G) {
        for i in 0..N {
            self.modules[i].update(grads);
        }
    }
}

impl<Input, T: Module<Input, Output = Input>, const N: usize> Module<Input> for Repeated<T, N> {
    type Output = T::Output;
    fn forward(&self, mut x: Input) -> Self::Output {
        for i in 0..N {
            x = self.modules[i].forward(x);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use rand::{prelude::StdRng, SeedableRng};

    #[test]
    fn test_default() {
        type Model = Repeated<(Linear<3, 3>, ReLU), 5>;
        let m: Model = Default::default();

        assert!(m.modules[0].0.weight.id < m.modules[1].0.weight.id);
        assert!(m.modules[1].0.weight.id < m.modules[2].0.weight.id);
        assert!(m.modules[2].0.weight.id < m.modules[3].0.weight.id);
        assert!(m.modules[3].0.weight.id < m.modules[4].0.weight.id);

        for i in 0..5 {
            assert_eq!(m.modules[i].0.weight.data(), &[[0.0; 3]; 3]);
            assert_eq!(m.modules[i].0.bias.data(), &[0.0; 3]);
        }
    }

    #[test]
    fn test_randomize() {
        type Model = Repeated<(Linear<3, 3>, ReLU), 5>;

        let mut rng = StdRng::seed_from_u64(0);
        let mut m: Model = Default::default();
        m.reset_params(&mut rng);

        for i in 0..5 {
            assert_ne!(m.modules[i].0.weight.data(), &[[0.0; 3]; 3]);
            assert_ne!(m.modules[i].0.bias.data(), &[0.0; 3]);
        }
    }

    #[test]
    fn test_forward() {
        type Model = Repeated<(Linear<3, 3>, ReLU), 5>;

        let mut rng = StdRng::seed_from_u64(0);
        let mut m: Model = Default::default();
        m.reset_params(&mut rng);

        let x = Tensor1D::zeros();
        let x = m.modules[0].forward(x);
        let x = m.modules[1].forward(x);
        let x = m.modules[2].forward(x);
        let x = m.modules[3].forward(x);
        let x = m.modules[4].forward(x);

        assert_eq!(x.data(), m.forward(Tensor1D::zeros()).data());
    }
}
