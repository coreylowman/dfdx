use crate::prelude::*;

/// A residual connection around `F`: `F(x) + x`,
/// as introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
///
/// # Generics
/// - `F`: The underlying module to do a skip connection around.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let module: ResidualAdd<ReLU> = Default::default();
/// let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = module.forward(x);
/// assert_eq!(y.data(), &[-2.0, -1.0, 0.0, 2.0, 4.0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ResidualAdd<F>(F);

impl<F: CanUpdateWithGradients> CanUpdateWithGradients for ResidualAdd<F> {
    /// Pass through to `F`'s [CanUpdateWithGradients].
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.0.update(grads);
    }
}

impl<F: ResetParams> ResetParams for ResidualAdd<F> {
    /// Pass through to `F`'s [ResetParams].
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.0.reset_params(rng);
    }
}

impl<T, F> Module<T> for ResidualAdd<F>
where
    T: Tensor<Dtype = f32>,
    F: Module<T, Output = T>,
{
    type Output = F::Output;

    /// Calls forward on `F` and then adds `x` to the result: `F(x) + x`
    fn forward(&self, x: T) -> Self::Output {
        let (x, tape) = x.split_tape();
        add(self.0.forward(x.duplicate().put_tape(tape)), &x)
    }
}

impl<F: SaveToNpz> SaveToNpz for ResidualAdd<F> {
    /// Pass through to `F`'s [SaveToNpz].
    fn write<W>(
        &self,
        filename_prefix: &str,
        w: &mut zip::ZipWriter<W>,
    ) -> zip::result::ZipResult<()>
    where
        W: std::io::Write + std::io::Seek,
    {
        self.0.write(filename_prefix, w)?;
        Ok(())
    }
}

impl<F: LoadFromNpz> LoadFromNpz for ResidualAdd<F> {
    /// Pass through to `F`'s [LoadFromNpz].
    fn read<R>(&mut self, filename_prefix: &str, r: &mut zip::ZipArchive<R>) -> Result<(), NpzError>
    where
        R: std::io::Read + std::io::Seek,
    {
        self.0.read(filename_prefix, r)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{prelude::StdRng, SeedableRng};
    use std::fs::File;
    use tempfile::NamedTempFile;
    use zip::ZipArchive;

    #[test]
    fn test_reset() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: ResidualAdd<Linear<2, 5>> = Default::default();
        assert_eq!(model.0.weight.data(), &[[0.0; 2]; 5]);
        assert_eq!(model.0.bias.data(), &[0.0; 5]);

        model.reset_params(&mut rng);
        assert_ne!(model.0.weight.data(), &[[0.0; 2]; 5]);
        assert_ne!(model.0.bias.data(), &[0.0; 5]);
    }

    const W0: [[f32; 2]; 5] = [
        [0.63315326, 0.3361526],
        [0.60201937, 0.30927354],
        [0.39831632, 0.29526848],
        [-0.4730785, -0.10664469],
        [0.5074884, -0.08458644],
    ];
    const B0: [f32; 5] = [-0.7014593, 0.01725882, 0.67181975, -0.61593556, 0.27809095];

    const W2: [[f32; 5]; 2] = [
        [0.37967658, -0.30938417, -0.4046409, 0.34131002, -0.36532],
        [0.01010674, 0.2922417, -0.28791183, 0.09316397, 0.00722069],
    ];
    const B2: [f32; 2] = [-0.01353309, 0.19437504];
    const X: [[f32; 2]; 10] = [
        [0.9706649, -0.50246257],
        [0.36609784, 0.22519696],
        [-0.26957038, -2.4395447],
        [0.729607, 0.06136635],
        [1.0758572, -0.6158074],
        [1.844528, -0.7769507],
        [-0.83232504, 0.26263165],
        [-0.18690403, 0.5396985],
        [-1.0891576, 0.09805013],
        [-0.63034505, 2.4173584],
    ];
    const Y: [[f32; 2]; 10] = [
        [0.15374291, -0.43383744],
        [-0.26277426, 0.25803787],
        [-0.41010314, -2.2426596],
        [-0.062764645, 0.117026225],
        [0.2237711, -0.54089284],
        [0.69048953, -0.6508272],
        [-1.0149324, 0.33670622],
        [-0.57907265, 0.53813595],
        [-1.2107061, 0.21556953],
        [-1.2221863, 2.3977249],
    ];

    const W0G: [[f32; 2]; 5] = [
        [0.035948314, -0.015142122],
        [-0.0035737813, -0.001155745],
        [-0.07784372, -0.059181444],
        [0.0, 0.0],
        [-0.081114516, 0.06281963],
    ];
    const B0G: [f32; 5] = [0.019489167, -0.005999865, -0.3116488, 0.0, -0.12533475];
    const W2G: [[f32; 5]; 2] = [[0.010261777, 0.15239798, 0.37232202, 0.0, 0.22712366]; 2];
    const B2G: [f32; 2] = [0.50000006; 2];

    #[test]
    fn test_residual_forward_and_backward() {
        type SubModel = (Linear<2, 5>, ReLU, Linear<5, 2>);
        type Model = ResidualAdd<SubModel>;

        let mut model: Model = Default::default();
        *model.0 .0.weight.mut_data() = W0;
        *model.0 .0.bias.mut_data() = B0;
        *model.0 .2.weight.mut_data() = W2;
        *model.0 .2.bias.mut_data() = B2;

        let x = Tensor2D::new(X);
        let y = model.forward(x.traced());
        assert_close(y.data(), &Y);

        let gradients = y.mean().backward();

        assert_close(gradients.ref_gradient(&model.0 .0.weight), &W0G);
        assert_close(gradients.ref_gradient(&model.0 .0.bias), &B0G);
        assert_close(gradients.ref_gradient(&model.0 .2.weight), &W2G);
        assert_close(gradients.ref_gradient(&model.0 .2.bias), &B2G);
    }

    #[test]
    fn test_save_residual() {
        let model: ResidualAdd<Linear<5, 3>> = Default::default();
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
    fn test_load_residual() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: ResidualAdd<Linear<5, 3>> = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: ResidualAdd<Linear<5, 3>> = Default::default();
        assert_ne!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_ne!(loaded_model.0.bias.data(), saved_model.0.bias.data());

        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        assert_eq!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_eq!(loaded_model.0.bias.data(), saved_model.0.bias.data());
    }
}
