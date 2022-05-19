use crate::prelude::*;
use rand::{distributions::Distribution, Rng};

/// A linear transformation of the form `xW + b`, where `W` is a matrix, `x` is a vector or matrix,
/// and `b` is a vector. If `x` is a matrix this does matrix multiplication.
///
/// Implements [Module] for both vectors of size `[I]` and batches of vectors of size `[B, I]`.
/// Implements [Randomize] to set weight & bias to random numbers drawn from a distribution.
///
/// Example usage:
/// `Linear<5, 2>` can act on vectors with 5 elements, and results in vectors with 2 elements.
/// ```
/// # use dfdx::prelude::*;
/// let model: Linear<5, 2> = Default::default();
/// assert_eq!(model.weight.data(), &[[0.0; 2]; 5]);
/// assert_eq!(model.bias.data(), &[0.0; 2]);
/// let x: Tensor1D<5> = Default::default();
/// let y: Tensor1D<2> = model.forward(x);
/// ```
#[derive(Default, Debug, Clone)]
pub struct Linear<const I: usize, const O: usize> {
    pub weight: Tensor2D<I, O, NoTape>,
    pub bias: Tensor1D<O, NoTape>,
}

impl<const I: usize, const O: usize> CanUpdateWithGradients for Linear<I, O> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.weight.update(grads);
        self.bias.update(grads);
    }
}

impl<const I: usize, const O: usize> Randomize for Linear<I, O> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        self.weight.randomize(rng, dist);
        self.bias.randomize(rng, dist);
    }
}

impl<const I: usize, const O: usize, H: TapeHolder> Module<Tensor1D<I, H>> for Linear<I, O> {
    type Output = Tensor1D<O, H>;

    /// 1d forward
    fn forward(&self, x: Tensor1D<I, H>) -> Self::Output {
        add(&self.bias, vecmat_mul(x, &self.weight))
    }
}

impl<const B: usize, const I: usize, const O: usize, H: TapeHolder> Module<Tensor2D<B, I, H>>
    for Linear<I, O>
{
    type Output = Tensor2D<B, O, H>;

    /// Batched 2d forward
    fn forward(&self, x: Tensor2D<B, I, H>) -> Self::Output {
        broadcast_outer_add(matmul(x, &self.weight), &self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: [[f32; 2]; 5] = [
        [-0.34588930, 0.11733949],
        [-0.30371523, 0.14059687],
        [-0.37120569, -0.10670426],
        [0.14303583, -0.09373143],
        [-0.02689660, 0.18974298],
    ];
    const B: [f32; 2] = [0.37653649, -0.29071701];

    #[test]
    fn test_forward_1d() {
        let model: Linear<5, 2> = Linear {
            weight: Tensor2D::new(W),
            bias: Tensor1D::new(B),
        };

        let x = Tensor1D::new([-0.88080013, 2.41853333, 2.24783349, 0.05652110, 2.03129911]);
        let y = model.forward(x);
        assert_eq!(y.data(), &[-0.93430865, 0.08624211]);
    }

    #[test]
    fn test_forward_2d() {
        let model: Linear<5, 2> = Linear {
            weight: Tensor2D::new(W),
            bias: Tensor1D::new(B),
        };

        let x = Tensor2D::new([
            [-1.94686651, 1.46117854, -1.66989815, 1.40886295, 1.34256434],
            [
                -1.33998311,
                3.05106783,
                -0.17936817,
                -0.04943254,
                -0.80527049,
            ],
            [
                -0.82914120,
                0.07691376,
                -0.26538327,
                0.90017676,
                -1.87904549,
            ],
        ]);
        let y = model.forward(x);
        assert_eq!(
            y.data(),
            &[
                [1.39143777, -0.012851536],
                [-0.005462587, -0.14800104],
                [0.9177769, -0.7897872]
            ]
        );
    }
}
