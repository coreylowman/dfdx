use crate::prelude::*;
use rand::Rng;
use rand_distr::Uniform;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

/// A linear transformation of the form `weight * x + bias`, where `weight` is a matrix, `x` is a vector or matrix,
/// and `bias` is a vector.
///
/// # Generics
/// - `I` The "input" size of vectors & matrices.
/// - `O` The "output" size of vectors & matrices.
///
/// # Examples
/// `Linear<5, 2>` can act on vectors with 5 elements, and results in vectors with 2 elements.
/// ```rust
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
    pub weight: Tensor2D<O, I, NoneTape>,

    /// Bias vector, shape (O, )
    pub bias: Tensor1D<O, NoneTape>,
}

impl<const I: usize, const O: usize> CanUpdateWithGradients for Linear<I, O> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.weight.update(grads);
        self.bias.update(grads);
    }
}

impl<const I: usize, const O: usize> ResetParams for Linear<I, O> {
    /// Initializes [Self::weight] and [Self::bias] from a [Uniform] distribution
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

impl<const I: usize, const O: usize> SaveToNpz for Linear<I, O> {
    /// Saves [Self::weight] to `{pre}weight.npy` and [Self::bias] to `{pre}bias.npy`
    /// using [npz_fwrite()].
    fn write<W>(&self, pre: &str, w: &mut ZipWriter<W>) -> ZipResult<()>
    where
        W: Write + Seek,
    {
        npz_fwrite(w, format!("{pre}weight.npy"), self.weight.data())?;
        npz_fwrite(w, format!("{pre}bias.npy"), self.bias.data())?;
        Ok(())
    }
}

impl<const I: usize, const O: usize> LoadFromNpz for Linear<I, O> {
    /// Reads [Self::weight] from `{pre}weight.npy` and [Self::bias] from `{pre}bias.npy`
    /// using [npz_fread()].
    fn read<R>(&mut self, pre: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError>
    where
        R: Read + Seek,
    {
        npz_fread(r, format!("{pre}weight.npy"), self.weight.mut_data())?;
        npz_fread(r, format!("{pre}bias.npy"), self.bias.mut_data())?;
        Ok(())
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

    /// Batched 2d forward using [matmul()] and [add_broadcast_rhs_first()]
    fn forward(&self, x: Tensor2D<B, I, H>) -> Self::Output {
        add_broadcast_rhs_first(matmul_transpose(x, &self.weight), &self.bias)
    }
}

#[cfg(test)]
mod tests {
    use rand::{prelude::StdRng, SeedableRng};
    use std::fs::File;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::tests::assert_close;

    const W: [[f32; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];
    const B: [f32; 2] = [0.3765365, -0.290717];

    #[test]
    fn test_forward_1d() {
        let model: Linear<5, 2> = Linear {
            weight: Tensor2D::new(W),
            bias: Tensor1D::new(B),
        };

        let x = Tensor1D::new([-0.8808001, 2.4185333, 2.2478335, 0.0565211, 2.031299]);
        let y = model.forward(x.trace());
        assert_close(y.data(), &[-0.93430865, 0.08624211]);

        let loss = y.square().mean();
        let gradients = loss.backward();
        assert_close(
            gradients.ref_gradient(&model.weight),
            &[
                [0.82293916, -2.2596567, -2.1001704, -0.05280815, -1.8978603],
                [-0.07596206, 0.20857942, 0.19385791, 0.004874499, 0.17518352],
            ],
        );
        assert_close(
            gradients.ref_gradient(&model.bias),
            &[-0.93430865, 0.08624211],
        );
    }

    #[test]
    fn test_forward_2d() {
        let model: Linear<5, 2> = Linear {
            weight: Tensor2D::new(W),
            bias: Tensor1D::new(B),
        };

        let x = Tensor2D::new([
            [-1.9468665, 1.4611785, -1.6698982, 1.408863, 1.3425643],
            [-1.3399831, 3.0510678, -0.17936817, -0.04943254, -0.8052705],
            [-0.8291412, 0.07691376, -0.26538327, 0.90017676, -1.8790455],
        ]);
        let y = model.forward(x.trace());
        assert_close(
            y.data(),
            &[
                [1.3914378, -0.012851536],
                [-0.005462587, -0.14800104],
                [0.9177769, -0.7897872],
            ],
        );

        let loss = y.square().mean();
        let gradients = loss.backward();
        assert_close(
            gradients.ref_gradient(&model.weight),
            &[
                [-1.1541969, 0.6956873, -0.8553807, 0.9289255, 0.04931633],
                [0.29272807, -0.17702839, 0.08586791, -0.24057935, 0.5286576],
            ],
        );
        assert_close(
            gradients.ref_gradient(&model.bias),
            &[0.7679174, -0.31687993],
        );
    }

    #[test]
    fn test_save_linear() {
        let model: Linear<5, 3> = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let mut zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        {
            let weight_file = zip
                .by_name("weight.npy")
                .expect("failed to find weight.npy file");
            assert!(weight_file.size() > 0);
        }
        {
            let bias_file = zip
                .by_name("bias.npy")
                .expect("failed to find bias.npy file");
            assert!(bias_file.size() > 0);
        }
    }

    #[test]
    fn test_load_linear() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: Linear<5, 3> = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: Linear<5, 3> = Default::default();
        assert!(loaded_model.weight.data() != saved_model.weight.data());
        assert!(loaded_model.bias.data() != saved_model.bias.data());

        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded_model.weight.data(), saved_model.weight.data());
        assert_eq!(loaded_model.bias.data(), saved_model.bias.data());
    }
}
