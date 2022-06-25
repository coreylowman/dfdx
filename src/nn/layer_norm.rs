use crate::prelude::*;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive};

/// Implements layer normalization as described in [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// This calls [normalize()] on the input to normalize to 0 mean and unit std dev, and then does an element-wise
/// affine transform using learnable parameters `self.gamma` and `self.beta`.
///
/// `self.epsilon` is passed to [normalize()] and added to the variance to ensure big enough numbers. It defaults to `1e-5`.
///
/// Generics:
/// - `M` The size of the affine transform tensors.
///
/// Implements:
/// - [Default] by setting `self.gamma` to all `1.0`s, and `self.beta` to all `0.0`s.
/// - [ResetParams] by setting `self.gamma` to all `1.0`s, and `self.beta` to all `0.0`s.
/// - [CanUpdateWithGradients]
/// - [Module<Tensor1D<M>>], which calls [normalize()], [mul()], and [add()].
/// - [Module<Tensor2D<B, M>>], which calls [normalize()], [mul_broadcast_rhs_first()], and [add_broadcast_rhs_first()].
/// - [SaveToNpz]
/// - [LoadFromNpz]
pub struct LayerNorm1D<const M: usize> {
    pub gamma: Tensor1D<M, NoTape>,
    pub beta: Tensor1D<M, NoTape>,
    pub epsilon: f32,
}

impl<const M: usize> Default for LayerNorm1D<M> {
    /// Fills `self.gamma` with 1s and `self.beta` with 0s and sets `self.epsilon` to `1e-5`.
    fn default() -> Self {
        Self {
            gamma: Tensor1D::ones(),
            beta: Tensor1D::zeros(),
            epsilon: 1e-5,
        }
    }
}

impl<const M: usize> ResetParams for LayerNorm1D<M> {
    /// Fills `self.gamma` with 1s and `self.beta` with 0s.
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {
        Cpu::fill(self.gamma.mut_data(), &mut || 1.0);
        Cpu::fill(self.beta.mut_data(), &mut || 0.0);
    }
}

impl<const M: usize> CanUpdateWithGradients for LayerNorm1D<M> {
    /// Updates `self.gamma` and `self.beta`.
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.gamma.update(grads);
        self.beta.update(grads);
    }
}

impl<H: Tape, const M: usize> Module<Tensor1D<M, H>> for LayerNorm1D<M> {
    type Output = Tensor1D<M, H>;

    /// Calls:
    /// 1. [normalize()] with `self.epsilon`
    /// 2. [mul()] with `self.gamma`
    /// 3. [add()] with `self.beta`
    fn forward(&self, x: Tensor1D<M, H>) -> Self::Output {
        let x = x.normalize(self.epsilon);
        let x = mul(x, &self.gamma);
        add(x, &self.beta)
    }
}

impl<H: Tape, const B: usize, const M: usize> Module<Tensor2D<B, M, H>> for LayerNorm1D<M> {
    type Output = Tensor2D<B, M, H>;

    /// Calls:
    /// 1. [normalize()] with `self.epsilon`.
    /// 2. [mul_broadcast_rhs_first()] with `self.gamma`
    /// 3. [add_broadcast_rhs_first()] with `self.beta`
    fn forward(&self, x: Tensor2D<B, M, H>) -> Self::Output {
        let x = x.normalize(self.epsilon);
        let x = mul_broadcast_rhs_first(x, &self.gamma);
        add_broadcast_rhs_first(x, &self.beta)
    }
}

impl<const M: usize> SaveToNpz for LayerNorm1D<M> {
    /// Saves `self.gamma` to `{pre}gamma.npy` and `self.beta` to `{pre}beta.npy`
    /// using [npz_fwrite()].
    fn write<W: Write + Seek>(&self, pre: &str, w: &mut zip::ZipWriter<W>) -> ZipResult<()> {
        npz_fwrite(w, format!("{pre}gamma.npy"), self.gamma.data())?;
        npz_fwrite(w, format!("{pre}beta.npy"), self.beta.data())?;
        Ok(())
    }
}

impl<const M: usize> LoadFromNpz for LayerNorm1D<M> {
    /// Reads `self.gamma` from `{p}gamma.npy` and `self.beta` from `{p}beta.npy`
    /// using [npz_fread()].
    fn read<R: Read + Seek>(&mut self, p: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
        npz_fread(r, format!("{p}gamma.npy"), self.gamma.mut_data())?;
        npz_fread(r, format!("{p}beta.npy"), self.beta.mut_data())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;
    use rand::{prelude::StdRng, SeedableRng};
    use rand_distr::Standard;
    use tempfile::NamedTempFile;

    #[test]
    fn test_layer_norm_reset() {
        let mut m: LayerNorm1D<5> = Default::default();
        assert_eq!(m.gamma.data(), &[1.0; 5]);
        assert_eq!(m.beta.data(), &[0.0; 5]);

        let mut rng = StdRng::seed_from_u64(0);
        m.gamma.randomize(&mut rng, &Standard);
        m.beta.randomize(&mut rng, &Standard);

        assert_ne!(m.gamma.data(), &[1.0; 5]);
        assert_ne!(m.beta.data(), &[0.0; 5]);

        m.reset_params(&mut rng);
        assert_eq!(m.gamma.data(), &[1.0; 5]);
        assert_eq!(m.beta.data(), &[0.0; 5]);
    }

    #[test]
    fn test_layer_norm_1d_forward() {
        let x = Tensor1D::new([
            -1.61233151,
            0.48965484,
            -1.57223654,
            -2.14012408,
            0.75928855,
            0.07052641,
            0.08577599,
            -0.94882685,
            -0.89430344,
            1.34884310,
        ]);
        let m: LayerNorm1D<10> = Default::default();
        let r = m.forward(x.trace());
        assert_eq!(
            r.data(),
            &[
                -1.0666381,
                0.84808326,
                -1.0301151,
                -1.5474098,
                1.0936954,
                0.4662948,
                0.48018575,
                -0.46224475,
                -0.4125788,
                1.6307268
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&m.gamma),
            &[
                -0.106663816,
                0.08480833,
                -0.10301151,
                -0.15474097,
                0.10936954,
                0.04662948,
                0.048018575,
                -0.046224475,
                -0.04125788,
                0.16307269
            ]
        );
        assert_eq!(gradients.ref_gradient(&m.beta), &[1.0 / 10.0; 10]);
    }

    #[test]
    fn test_layer_norm_2d_forward() {
        let x = Tensor2D::new(X_2);
        let m: LayerNorm1D<10> = Default::default();
        let r = m.forward(x.trace());
        assert_eq!(r.data(), &Y_2);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&m.gamma),
            &[
                -0.020472767,
                0.009951305,
                0.021753367,
                0.07825497,
                -0.064731464,
                0.02485959,
                -0.0044674473,
                -0.06997709,
                0.002105412,
                0.022724135
            ]
        );
        assert_eq!(gradients.ref_gradient(&m.beta), &[0.099999994; 10]);
    }

    #[test]
    fn test_save_layer_norm() {
        let model: LayerNorm1D<13> = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap().to_string())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        let mut names = zip.file_names().collect::<Vec<&str>>();
        names.sort();
        assert_eq!(&names, &["beta.npy", "gamma.npy",]);
    }

    #[test]
    fn test_save_layer_norm_tuple() {
        let model: (LayerNorm1D<5>, LayerNorm1D<13>) = Default::default();
        let file = NamedTempFile::new().expect("failed to create tempfile");
        model
            .save(file.path().to_str().unwrap().to_string())
            .expect("failed to save model");
        let f = File::open(file.path()).expect("failed to open resulting file");
        let zip = ZipArchive::new(f).expect("failed to create zip archive from file");
        let mut names = zip.file_names().collect::<Vec<&str>>();
        names.sort();
        assert_eq!(
            &names,
            &["0.beta.npy", "0.gamma.npy", "1.beta.npy", "1.gamma.npy"]
        );
    }

    #[test]
    fn test_load_layer_norm() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: LayerNorm1D<13> = Default::default();
        saved_model.gamma.randomize(&mut rng, &Standard);
        saved_model.beta.randomize(&mut rng, &Standard);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model
            .save(file.path().to_str().unwrap().to_string())
            .is_ok());

        let mut loaded_model: LayerNorm1D<13> = Default::default();
        assert!(loaded_model.gamma.data() != saved_model.gamma.data());
        assert!(loaded_model.beta.data() != saved_model.beta.data());

        assert!(loaded_model
            .load(file.path().to_str().unwrap().to_string())
            .is_ok());
        assert_eq!(loaded_model.gamma.data(), saved_model.gamma.data());
        assert_eq!(loaded_model.beta.data(), saved_model.beta.data());
    }

    const X_2: [[f32; 10]; 5] = [
        [
            0.29491714,
            -0.23289900,
            -0.28846350,
            -0.77137190,
            -0.46175328,
            -0.64002252,
            -0.35834178,
            -1.54459560,
            -1.48547590,
            0.94435787,
        ],
        [
            -1.99742687,
            1.75386345,
            -0.00747265,
            0.70845670,
            0.37745902,
            1.24608839,
            -0.55608803,
            -1.75963795,
            -0.37871835,
            -0.95974267,
        ],
        [
            0.80032063,
            -1.23041463,
            1.16585624,
            1.27445364,
            0.57286376,
            0.57976252,
            0.67744941,
            0.05471262,
            0.12386650,
            -0.73329753,
        ],
        [
            -1.65339887,
            0.24153176,
            0.41823727,
            2.09267616,
            -2.05458617,
            0.44928759,
            -0.04141246,
            -1.74209344,
            2.40003014,
            1.89292789,
        ],
        [
            0.03640575,
            0.39946404,
            -0.41127914,
            0.82208872,
            -1.91227925,
            -0.16858509,
            -0.26039550,
            0.75304174,
            0.42073044,
            0.08859433,
        ],
    ];

    const Y_2: [[f32; 10]; 5] = [
        [
            1.0606476,
            0.31349644,
            0.23484199,
            -0.44873992,
            -0.010458682,
            -0.26280802,
            0.13592565,
            -1.5432781,
            -1.4595912,
            1.9799646,
        ],
        [
            -1.5845544, 1.6457634, 0.12903845, 0.7455408, 0.4605115, 1.2085072, -0.3433862,
            -1.3797892, -0.1906493, -0.690982,
        ],
        [
            0.62280494,
            -2.0580986,
            1.105372,
            1.2487383,
            0.3225246,
            0.3316321,
            0.4605948,
            -0.36151984,
            -0.2702254,
            -1.4018224,
        ],
        [
            -1.2028971,
            0.026742745,
            0.14140874,
            1.2279692,
            -1.4632316,
            0.16155761,
            -0.15686263,
            -1.2604518,
            1.4274143,
            1.0983504,
        ],
        [
            0.08036063,
            0.56966114,
            -0.52299273,
            1.1392404,
            -2.545919,
            -0.19590938,
            -0.319644,
            1.0461844,
            0.5983223,
            0.15069617,
        ],
    ];
}
