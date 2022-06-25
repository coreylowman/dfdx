use crate::prelude::*;
use rand::prelude::Rng;
use std::io::{Read, Seek, Write};
use zip::{result::ZipResult, ZipArchive, ZipWriter};

impl<Input, A, B> Module<Input> for (A, B)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
{
    type Output = B::Output;
    fn forward(&self, x: Input) -> Self::Output {
        let x = self.0.forward(x);
        self.1.forward(x)
    }
}

impl<Input, A, B, C> Module<Input> for (A, B, C)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
    C: Module<B::Output>,
{
    type Output = C::Output;
    fn forward(&self, x: Input) -> Self::Output {
        let x = self.0.forward(x);
        let x = self.1.forward(x);
        self.2.forward(x)
    }
}

impl<Input, A, B, C, D> Module<Input> for (A, B, C, D)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
    C: Module<B::Output>,
    D: Module<C::Output>,
{
    type Output = D::Output;
    fn forward(&self, x: Input) -> Self::Output {
        let x = self.0.forward(x);
        let x = self.1.forward(x);
        let x = self.2.forward(x);
        self.3.forward(x)
    }
}

impl<Input, A, B, C, D, E> Module<Input> for (A, B, C, D, E)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
    C: Module<B::Output>,
    D: Module<C::Output>,
    E: Module<D::Output>,
{
    type Output = E::Output;
    fn forward(&self, x: Input) -> Self::Output {
        let x = self.0.forward(x);
        let x = self.1.forward(x);
        let x = self.2.forward(x);
        let x = self.3.forward(x);
        self.4.forward(x)
    }
}

impl<Input, A, B, C, D, E, F> Module<Input> for (A, B, C, D, E, F)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
    C: Module<B::Output>,
    D: Module<C::Output>,
    E: Module<D::Output>,
    F: Module<E::Output>,
{
    type Output = F::Output;
    fn forward(&self, x: Input) -> Self::Output {
        let x = self.0.forward(x);
        let x = self.1.forward(x);
        let x = self.2.forward(x);
        let x = self.3.forward(x);
        let x = self.4.forward(x);
        self.5.forward(x)
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+]) => {
        impl<$($name: CanUpdateWithGradients),+> CanUpdateWithGradients for ($($name,)+) {
            fn update<G: GradientProvider>(&mut self, grads: &mut G) {
                $(self.$idx.update(grads));+
            }
        }

        impl<$($name: ResetParams),+> ResetParams for ($($name,)+) {
            fn reset_params<R: Rng>(&mut self, rng: &mut R) {
                $(self.$idx.reset_params(rng));+
            }
        }

        impl<$($name: SaveToNpz),+> SaveToNpz for ($($name,)+) {
            /// Calls `SaveToNpz::write(self.<idx>, ...)` on each part of the tuple. See [SaveToNpz].
            ///
            /// E.g. for a two tuple (A, B) with `base == ""`, this will call:
            /// 1. `self.0.write("0.", w)`
            /// 2. `self.1.write("1.", w)`
            fn write<W: Write + Seek>(&self, base: &str, w: &mut ZipWriter<W>) -> ZipResult<()> {
                $(self.$idx.write(&format!("{}{}.", base, $idx), w)?;)+
                Ok(())
            }
        }

        impl<$($name: LoadFromNpz),+> LoadFromNpz for ($($name,)+) {
            /// Calls `LoadFromNpz::read(self.<idx>, ...)` on each part of the tuple. See [LoadFromNpz].
            ///
            /// E.g. for a two tuple (A, B) with `base == ""`, this will call:
            /// 1. `self.0.read("0.", r)`
            /// 2. `self.1.read("1.", r)`
            fn read<R: Read + Seek>(&mut self, base: &str, r: &mut ZipArchive<R>) -> Result<(), NpzError> {
                $(self.$idx.read(&format!("{}{}.", base, $idx), r)?;)+
                Ok(())
            }
        }
    };
}

tuple_impls!([A, B] [0, 1]);
tuple_impls!([A, B, C] [0, 1, 2]);
tuple_impls!([A, B, C, D] [0, 1, 2, 3]);
tuple_impls!([A, B, C, D, E] [0, 1, 2, 3, 4]);
tuple_impls!([A, B, C, D, E, F] [0, 1, 2, 3, 4, 5]);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::StdRng, SeedableRng};
    use std::fs::File;
    use tempfile::NamedTempFile;

    #[test]
    fn test_2_tuple() {
        let model: (ReLU, Tanh) = Default::default();

        let x = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = model.forward(x);
        assert_eq!(y.data(), &[0.0, 0.0, 0.0, 1.0f32.tanh(), 2.0f32.tanh()]);
    }

    #[test]
    fn test_2_tuple_update() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: (Linear<2, 3>, Linear<3, 4>) = Default::default();
        model.reset_params(&mut rng);
        assert!(model.0.weight.data() != &[[0.0; 2]; 3]);
        assert!(model.0.bias.data() != &[0.0; 3]);
        assert!(model.1.weight.data() != &[[0.0; 3]; 4]);
        assert!(model.1.bias.data() != &[0.0; 4]);

        let m0 = model.clone();

        let loss = model
            .forward(Tensor1D::randn(&mut rng).traced())
            .square()
            .mean();
        let gradients = loss.backward();

        assert!(gradients.ref_gradient(&model.0.weight) != &[[0.0; 2]; 3]);
        assert!(gradients.ref_gradient(&model.0.bias) != &[0.0; 3]);
        assert!(gradients.ref_gradient(&model.1.weight) != &[[0.0; 3]; 4]);
        assert!(gradients.ref_gradient(&model.1.bias) != &[0.0; 4]);

        let mut sgd = Sgd::new(1.0, None);
        sgd.update(&mut model, gradients);

        assert!(model.0.weight.data() != m0.0.weight.data());
        assert!(model.0.bias.data() != m0.0.bias.data());
        assert!(model.1.weight.data() != m0.1.weight.data());
        assert!(model.1.bias.data() != m0.1.bias.data());
    }

    #[test]
    fn test_save_tuple() {
        let model: (
            Linear<1, 2>,
            ReLU,
            Linear<2, 3>,
            (Dropout, Linear<1, 2>, Linear<3, 4>),
        ) = Default::default();
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
            &[
                "0.bias.npy",
                "0.weight.npy",
                "2.bias.npy",
                "2.weight.npy",
                "3.1.bias.npy",
                "3.1.weight.npy",
                "3.2.bias.npy",
                "3.2.weight.npy",
            ]
        );
    }

    #[test]
    fn test_load_tuple() {
        type Model = (
            Linear<1, 2>,
            ReLU,
            Linear<2, 3>,
            (Dropout, Linear<1, 2>, Linear<3, 4>),
        );

        let mut rng = StdRng::seed_from_u64(0);
        let mut saved_model: Model = Default::default();
        saved_model.reset_params(&mut rng);

        let file = NamedTempFile::new().expect("failed to create tempfile");
        assert!(saved_model
            .save(file.path().to_str().unwrap().to_string())
            .is_ok());

        let mut loaded_model: Model = Default::default();
        assert!(loaded_model
            .load(file.path().to_str().unwrap().to_string())
            .is_ok());
        assert_eq!(loaded_model.0.weight.data(), saved_model.0.weight.data());
        assert_eq!(loaded_model.0.bias.data(), saved_model.0.bias.data());
        assert_eq!(loaded_model.2.weight.data(), saved_model.2.weight.data());
        assert_eq!(loaded_model.2.bias.data(), saved_model.2.bias.data());
        assert_eq!(
            loaded_model.3 .1.weight.data(),
            saved_model.3 .1.weight.data()
        );
        assert_eq!(loaded_model.3 .1.bias.data(), saved_model.3 .1.bias.data());

        assert_eq!(
            loaded_model.3 .2.weight.data(),
            saved_model.3 .2.weight.data()
        );
        assert_eq!(loaded_model.3 .2.bias.data(), saved_model.3 .2.bias.data());
    }
}
