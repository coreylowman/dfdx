use crate::prelude::*;

/// A residual connection `R` around `F`: `F(x) + R(x)`,
/// as introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
///
/// # Generics
/// - `F`: The underlying module to do a skip connection around.
/// - `R`: The underlying residual module
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let module: Residual<ReLU, Square> = Default::default();
/// let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let y = module.forward(x);
/// assert_eq!(y.data(), &[4.0, 1.0, 0.0, 2.0, 6.0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Residual<F, R>(F, R);

impl<F: CanUpdateWithGradients, R: CanUpdateWithGradients> CanUpdateWithGradients
    for Residual<F, R>
{
    /// Pass through to `F`'s [CanUpdateWithGradients].
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.0.update(grads);
        self.1.update(grads);
    }
}

impl<F: ResetParams, R: ResetParams> ResetParams for Residual<F, R> {
    /// Pass through to `F`'s [ResetParams].
    fn reset_params<RNG: rand::Rng>(&mut self, rng: &mut RNG) {
        self.0.reset_params(rng);
        self.1.reset_params(rng);
    }
}

impl<O, T, F, R> Module<T> for Residual<F, R>
where
    O: Tensor<Dtype = f32>,
    T: Tensor<Dtype = f32, Tape = O::Tape>,
    F: Module<T, Output = O>,
    R: Module<T, Output = O>,
{
    type Output = O;

    /// Calls forward on `F` and `R` and then sums their result: `F(x) + R(x)`
    fn forward(&self, x: T) -> Self::Output {
        let (x, mut tape) = x.split_tape();
        // creating 2 new tensors with different ids but same data (of x) and then sum their f(x) and f'(x)
        let mut main_input: Box<T::Array> = T::Device::zeros();
        main_input.as_mut().clone_from(x.data());
        let main_input: T::NoTape = TensorCreator::new_boxed(main_input);
        let main_input_phantom = main_input.phantom();

        let mut residual_input: Box<T::Array> = T::Device::zeros();
        residual_input.as_mut().clone_from(x.data());
        let residual_input: T::NoTape = TensorCreator::new_boxed(residual_input);
        let residual_input_phantom = residual_input.phantom();
        let input_phantom = x.phantom();

        // sum their derivatives
        tape.add_backward_op(move |grads| {
            {
                let (total_grad, main_grad) =
                    grads.mut_and_ref(&input_phantom, &main_input_phantom);
                total_grad.clone_from(main_grad);
            }
            let (total_grad, res_grad) = grads.mut_and_ref(&input_phantom, &residual_input_phantom);
            T::Device::add(total_grad, &res_grad);
        });

        // F(x)
        let (main, tape) = self.0.forward(main_input.put_tape(tape)).split_tape();
        let main_output = main.phantom();
        // R(x)
        let (residual, mut tape) = self.1.forward(residual_input.put_tape(tape)).split_tape();
        let residual_output = residual.phantom();
        // copy data from F(x) back to a new x (since there might be a new shape) and then sum it with R(x)
        let mut new_x: Box<O::Array> = O::Device::zeros();
        new_x.as_mut().clone_from(main.data());
        let mut new_x: O::NoTape = TensorCreator::new_boxed(new_x);
        O::Device::add(new_x.mut_data(), residual.data());

        // both F'(x) and R'(x) have the same 'starting gradient'
        // => copy the 'starting gradient' into F'(x)/R'(x)'s output tensor's gradient
        let result = new_x.phantom();
        tape.add_backward_op(move |grads| {
            {
                let (main_grad, result_grad) = grads.mut_and_ref(&main_output, &result);
                main_grad.clone_from(result_grad);
            }
            let (res_grad, result_grad) = grads.mut_and_ref(&residual_output, &result);
            res_grad.clone_from(result_grad);
        });
        new_x.put_tape(tape)
    }
}

impl<F: SaveToNpz, R: SaveToNpz> SaveToNpz for Residual<F, R> {
    /// Pass through to `F`/`R`'s [SaveToNpz].
    fn write<W>(
        &self,
        filename_prefix: &str,
        w: &mut zip::ZipWriter<W>,
    ) -> zip::result::ZipResult<()>
    where
        W: std::io::Write + std::io::Seek,
    {
        // I have no idea what to put here
        self.0.write(filename_prefix, w)?;
        self.1.write(filename_prefix, w)?;
        Ok(())
    }
}

impl<F: LoadFromNpz, R: LoadFromNpz> LoadFromNpz for Residual<F, R> {
    /// Pass through to `F`/`R`'s [LoadFromNpz].
    fn read<READ>(
        &mut self,
        filename_prefix: &str,
        r: &mut zip::ZipArchive<READ>,
    ) -> Result<(), NpzError>
    where
        READ: std::io::Read + std::io::Seek,
    {
        // I have no idea what to put here, reverse order since it's like a stack?
        self.1.read(filename_prefix, r)?;
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
        let mut model: Residual<Linear<2, 5>, Linear<2, 5>> = Default::default();
        assert_eq!(model.0.weight.data(), &[[0.0; 2]; 5]);
        assert_eq!(model.0.bias.data(), &[0.0; 5]);
        assert_eq!(model.1.weight.data(), &[[0.0; 2]; 5]);
        assert_eq!(model.1.bias.data(), &[0.0; 5]);

        model.reset_params(&mut rng);
        assert_ne!(model.0.weight.data(), &[[0.0; 2]; 5]);
        assert_ne!(model.0.bias.data(), &[0.0; 5]);
        assert_ne!(model.1.weight.data(), &[[0.0; 2]; 5]);
        assert_ne!(model.1.bias.data(), &[0.0; 5]);
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
    fn test_residual_forward_backward_resadd_as_main() {
        type SubModel = (Linear<2, 5>, ReLU, Linear<5, 2>);
        type Model = Residual<SubModel, ReLU>;

        let mut model: Model = Default::default();
        *model.0 .0.weight.mut_data() = W0;
        *model.0 .0.bias.mut_data() = B0;
        *model.0 .2.weight.mut_data() = W2;
        *model.0 .2.bias.mut_data() = B2;

        let x = Tensor2D::new(X);
        let y = model.forward(x.traced());
        // Y = s(x) + x, including negative x
        // Y2 would be s(x) + r(x) [r == ReLU]
        // Y2 = s(x) + r(x) = Y - x + r(x)
        // Y2 = Y - (x - r(x))
        // x - r(x) = {0, if x >= 0, because r(x) = x
        // x - r(x) = {x, if x < 0, because r(x) = 0 => x - 0 = x
        // this is r(-x), since this returns x if x < 0 and 0 elsewhere
        // => Y2 = Y - r(-x)
        assert_close(
            y.data(),
            add(Tensor2D::new(Y), &(-Tensor2D::new(X)).relu()).data(),
        );

        let gradients = y.mean().backward();

        assert_close(gradients.ref_gradient(&model.0 .0.weight), &W0G);
        assert_close(gradients.ref_gradient(&model.0 .0.bias), &B0G);
        assert_close(gradients.ref_gradient(&model.0 .2.weight), &W2G);
        assert_close(gradients.ref_gradient(&model.0 .2.bias), &B2G);
    }

    #[test]
    fn test_residual_forward_backward_with_update() {
        type SubModel = (Linear<2, 5>, ReLU, Linear<5, 2>);
        type Model = Residual<SubModel, SubModel>;

        let mut model: Model = Default::default();
        *model.0 .0.weight.mut_data() = W0;
        *model.0 .0.bias.mut_data() = B0;
        *model.0 .2.weight.mut_data() = W2;
        *model.0 .2.bias.mut_data() = B2;
        *model.1 .0.weight.mut_data() = W0;
        *model.1 .0.bias.mut_data() = B0;
        *model.1 .2.weight.mut_data() = W2;
        *model.1 .2.bias.mut_data() = B2;

        let mut model2: SubModel = Default::default();
        *model2.0.weight.mut_data() = W0;
        *model2.0.bias.mut_data() = B0;
        // The submodel s(x) = l(x) with l(x) = ax + b and is the last linear layer
        // model2 has to be model + model = 2 * model => s2(x) = 2 * s(x) => s2(x) = 2ax + 2b
        // => a' = 2a; b' = 2b
        *model2.2.weight.mut_data() = W2;
        model2.2.weight = model2.2.weight * 2.0;
        *model2.2.bias.mut_data() = B2;
        model2.2.bias = model2.2.bias * 2.0;

        let x = Tensor2D::new(X);
        let y = model.forward(x.traced());
        let x2 = Tensor2D::new(X);
        let y2 = model2.forward(x2.traced());
        assert_close(y.data(), y2.data());

        let gradients = y.mean().backward();
        let gradients2 = y2.mean().backward();

        assert_close(gradients.ref_gradient(&model.0 .0.weight), &W0G);
        assert_close(gradients.ref_gradient(&model.0 .0.bias), &B0G);
        assert_close(gradients.ref_gradient(&model.0 .2.weight), &W2G);
        assert_close(gradients.ref_gradient(&model.0 .2.bias), &B2G);
        assert_close(gradients.ref_gradient(&model.1 .0.weight), &W0G);
        assert_close(gradients.ref_gradient(&model.1 .0.bias), &B0G);
        assert_close(gradients.ref_gradient(&model.1 .2.weight), &W2G);
        assert_close(gradients.ref_gradient(&model.1 .2.bias), &B2G);
        assert_close(
            gradients2.ref_gradient(&model2.0.weight),
            (Tensor2D::new(W0G) * 2.0).data(),
        );
        assert_close(
            gradients2.ref_gradient(&model2.0.bias),
            (Tensor1D::new(B0G) * 2.0).data(),
        );
        // no multiplication with 2 here since f'(x) = h'(x) * g'(h(j(x))) with f(x) = g(h(j(x)))
        // In this case, it's f(x) = g(h(j(2x))) => f'(x) = h'(j(2x)) * g'(h(j(x))),
        // while g(x) = h(j(2x)) => g'(x) = 2 * j'(x) * h'(j(x))
        assert_close(gradients2.ref_gradient(&model2.2.weight), &W2G);
        assert_close(gradients2.ref_gradient(&model2.2.bias), &B2G);

        // with lr = 1, w* = w - w'
        let sgd_config = SgdConfig {
            lr: 1.0,
            momentum: None,
        };
        Sgd::new(sgd_config).update(&mut model, gradients);
        Sgd::new(sgd_config).update(&mut model2, gradients2);

        assert_close(
            model.0 .0.weight.data(),
            sub(Tensor2D::new(W0), &Tensor2D::new(W0G)).data(),
        );
        assert_close(
            model.0 .0.bias.data(),
            sub(Tensor1D::new(B0), &Tensor1D::new(B0G)).data(),
        );
        assert_close(
            model.0 .2.weight.data(),
            sub(Tensor2D::new(W2), &Tensor2D::new(W2G)).data(),
        );
        assert_close(
            model.0 .2.bias.data(),
            sub(Tensor1D::new(B2), &Tensor1D::new(B2G)).data(),
        );
        assert_close(
            model.1 .0.weight.data(),
            sub(Tensor2D::new(W0), &Tensor2D::new(W0G)).data(),
        );
        assert_close(
            model.1 .0.bias.data(),
            sub(Tensor1D::new(B0), &Tensor1D::new(B0G)).data(),
        );
        assert_close(
            model.1 .2.weight.data(),
            sub(Tensor2D::new(W2), &Tensor2D::new(W2G)).data(),
        );
        assert_close(
            model.1 .2.bias.data(),
            sub(Tensor1D::new(B2), &Tensor1D::new(B2G)).data(),
        );
    }

    // gradients have to be summed, r(x) = g(x) + h(x) => r'(x) = g'(x) + h'(x)
    #[test]
    fn test_residual_gradients_correctly_added() {
        type Model = (Linear<1, 1>, Residual<ReLU, ReLU>);
        // Linear<1, 2>-layer has weights with one and bias zeroed
        let mut model: Model = Default::default();
        *model.0.weight.mut_data() = [[1.0]];

        let x = Tensor2D::new([[-1.0], [1.0]]);
        let y = model.forward(x.traced());

        assert_close(y.data(), &[[0.0], [2.0]]);

        let grads = y.mean().backward();

        // m(x) = r(x) + r(x) = 2r(x); m'(x) = 2r'(x)
        // y_mean(x_1, x_2) = (m(x_1) + m(x_2)) / 2 = (2r(x_1) + 2r(x_2)) / 2 = r(x_1) + r(x_2)
        // x_1 = -1; x_2 = 1 => r(-1) + r(1) = 0 + 1 = 1
        // y_mean'(x_1, x_2) = r'(x_1) + r'(x_2) = r'(-1) + r'(1) = 0 + 1 = 1
        // x_1 and x_2 are from the Linear layer, so x_1 = ax + b with a = 1 and b = 0
        // => derivates for a and b are the same
        assert_close(grads.ref_gradient(&model.0.weight), &[[1.0]]);
        assert_close(grads.ref_gradient(&model.0.bias), &[1.0]);
    }

    // test for different input and output shapes?

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

    /// TODO test fails
    #[test]
    fn test_load_residual() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: Residual<Linear<5, 3>, Linear<5, 3>> = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model.save(file.path().to_str().unwrap()).is_ok());

        let mut loaded_model: Residual<Linear<5, 3>, Linear<5, 3>> = Default::default();
        assert_ne!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_ne!(loaded_model.0.bias.data(), saved_model.0.bias.data());

        assert_ne!(loaded_model.1.weight.data(), saved_model.1.weight.data());
        assert_ne!(loaded_model.1.bias.data(), saved_model.1.bias.data());

        assert!(loaded_model.load(file.path().to_str().unwrap()).is_ok());
        // only the next 2 lines are 'failing
        assert_eq!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_eq!(loaded_model.0.bias.data(), saved_model.0.bias.data());

        assert_eq!(loaded_model.1.weight.data(), saved_model.1.weight.data());
        assert_eq!(loaded_model.1.bias.data(), saved_model.1.bias.data());
    }
}
