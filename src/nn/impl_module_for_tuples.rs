use crate::prelude::*;
use rand::prelude::{Distribution, Rng};

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

        impl<$($name: Randomize<f32>),+> Randomize<f32> for ($($name,)+) {
            fn randomize<R: Rng, DIST: Distribution<f32>>(&mut self, rng: &mut R, dist: &DIST) {
                $(self.$idx.randomize(rng, dist));+
            }
        }

        impl<$($name: SaveToZip),+> SaveToZip for ($($name,)+) {
            /// Calls `SaveToZip::write(self.<idx>, ...)` on each part of the tuple.
            fn write<W>(&self, base: &String, w: &mut zip::ZipWriter<W>) -> zip::result::ZipResult<()>
            where
                W: std::io::Write + std::io::Seek
            {
                $(self.$idx.write(&format!("{}{}.", base, $idx), w)?;)+
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
    use rand::{prelude::StdRng, SeedableRng};
    use rand_distr::StandardNormal;
    use std::fs::File;
    use tempfile::NamedTempFile;
    use zip::ZipArchive;

    use super::*;

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
        model.randomize(&mut rng, &StandardNormal);
        assert!(model.0.weight.data() != &[[0.0; 3]; 2]);
        assert!(model.0.bias.data() != &[0.0; 3]);
        assert!(model.1.weight.data() != &[[0.0; 4]; 3]);
        assert!(model.1.bias.data() != &[0.0; 4]);

        let m0 = model.clone();

        let loss = model
            .forward(Tensor1D::randn(&mut rng).traced())
            .square()
            .mean();
        let gradients = loss.backward();

        assert!(gradients.ref_gradient(&model.0.weight) != &[[0.0; 3]; 2]);
        assert!(gradients.ref_gradient(&model.0.bias) != &[0.0; 3]);
        assert!(gradients.ref_gradient(&model.1.weight) != &[[0.0; 4]; 3]);
        assert!(gradients.ref_gradient(&model.1.bias) != &[0.0; 4]);

        let mut sgd = Sgd::new(1.0, None);
        sgd.update(&mut model, gradients);

        assert!(model.0.weight.data() != m0.0.weight.data());
        assert!(model.0.bias.data() != m0.0.bias.data());
        assert!(model.1.weight.data() != m0.1.weight.data());
        assert!(model.1.bias.data() != m0.1.bias.data());
    }

    #[test]
    fn test_save() {
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
}
