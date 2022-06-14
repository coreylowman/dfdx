use crate::prelude::*;
use rand::Rng;
use rand_distr::Uniform;

/// A linear transformation of the form `x * transpose(W) + b`, where `W` is a matrix, `x` is a vector or matrix,
/// and `b` is a vector. If `x` is a matrix this does matrix multiplication.
///
/// Implements:
/// - [Module] for vectors like [Tensor1D<I>]
/// - [Module] for matrices like [Tensor2D<B, I>], where `B` is batch size
/// - [ResetParams] to set weight & bias to uniform random numbers from a distribution based on `I`.
/// - [CanUpdateWithGradients]
/// - [SaveToNpz]
/// - [LoadFromNpz]
///
/// Generics:
/// - `I` The input size of vectors & matrices.
/// - `O` The output size of vectors & matrices.
///
/// Example usage:
/// `Linear<5, 2>` can act on vectors with 5 elements, and results in vectors with 2 elements.
/// ```
/// # use dfdx::prelude::*;
/// let model: Linear<5, 2> = Default::default();
/// assert_eq!(model.weight.data(), &[[0.0; 5]; 2]);
/// assert_eq!(model.bias.data(), &[0.0; 2]);
/// let x: Tensor1D<5> = Default::default();
/// let y: Tensor1D<2> = model.forward(x);
/// assert_eq!(y.data(), &[0.0; 2]);
/// ```
#[derive(Default, Debug, Clone)]
pub struct Linear<const I: usize, const O: usize> {
    /// Transposed weight matrix, shape (O, I)
    pub weight: Tensor2D<O, I, NoTape>,

    /// Bias vector, shape (O, )
    pub bias: Tensor1D<O, NoTape>,
}

impl<const I: usize, const O: usize> CanUpdateWithGradients for Linear<I, O> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.weight.update(grads);
        self.bias.update(grads);
    }
}

impl<const I: usize, const O: usize> ResetParams for Linear<I, O> {
    /// Initializes `self.weight` and `self.bias` from a [Uniform] distribution
    /// between [-1 / sqrt(I), 1 / sqrt(I)].
    ///
    /// This uses [Randomize::randomize()] to set the values of the tensor.
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        let bound: f32 = 1.0 / (I as f32).sqrt();
        let dist = Uniform::new(-bound, bound);
        self.weight.randomize(rng, &dist);
        self.bias.randomize(rng, &dist);
    }
}

impl<const I: usize, const O: usize, H: Tape> Module<Tensor1D<I, H>> for Linear<I, O> {
    type Output = Tensor1D<O, H>;

    /// 1d forward using [vecmat_mul()] and [add()].
    fn forward(&self, x: Tensor1D<I, H>) -> Self::Output {
        add(vecmat_mul_transpose(x, &self.weight), &self.bias)
    }
}

impl<const B: usize, const I: usize, const O: usize, H: Tape> Module<Tensor2D<B, I, H>>
    for Linear<I, O>
{
    type Output = Tensor2D<B, O, H>;

    /// Batched 2d forward using [matmul()] and [broadcast_outer_add()]
    fn forward(&self, x: Tensor2D<B, I, H>) -> Self::Output {
        broadcast_outer_add(matmul_transpose(x, &self.weight), &self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: [[f32; 5]; 2] = [
        [
            -0.34588930,
            -0.30371523,
            -0.37120569,
            0.14303583,
            -0.02689660,
        ],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];
    const B: [f32; 2] = [0.37653649, -0.29071701];

    #[test]
    fn test_forward_1d() {
        let model: Linear<5, 2> = Linear {
            weight: Tensor2D::new(W),
            bias: Tensor1D::new(B),
        };

        let x = Tensor1D::new([-0.88080013, 2.41853333, 2.24783349, 0.05652110, 2.03129911]);
        let y = model.forward(x.trace());
        assert_eq!(y.data(), &[-0.93430865, 0.08624211]);

        let loss = y.square().mean();
        let gradients = loss.backward();
        assert_eq!(
            gradients.ref_gradient(&model.weight),
            &[
                [
                    0.82293916,
                    -2.25965667,
                    -2.10017037,
                    -0.05280815,
                    -1.89786029,
                ],
                [-0.07596206, 0.20857942, 0.19385791, 0.004874499, 0.17518352],
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&model.bias),
            &[-0.93430865, 0.08624211]
        );
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
        let y = model.forward(x.trace());
        assert_eq!(
            y.data(),
            &[
                [1.39143777, -0.012851536],
                [-0.005462587, -0.14800104],
                [0.9177769, -0.7897872]
            ]
        );

        let loss = y.square().mean();
        let gradients = loss.backward();
        assert_eq!(
            gradients.ref_gradient(&model.weight),
            &[
                [-1.15419686, 0.69568729, -0.85538071, 0.92892551, 0.04931633],
                [0.29272807, -0.17702839, 0.08586791, -0.24057935, 0.52865762],
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&model.bias),
            &[0.76791739, -0.31687993]
        );
    }
}
