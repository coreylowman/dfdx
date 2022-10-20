use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;
use rand::prelude::Rng;

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+], $last:ident, [$($rev_tail:ident),+]) => {
        impl<$($name: CanUpdateWithGradients),+> CanUpdateWithGradients for ($($name,)+) {
            fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
                $(self.$idx.update(grads, unused);)+
            }
        }

        impl<$($name: ResetParams),+> ResetParams for ($($name,)+) {
            fn reset_params<R: Rng>(&mut self, rng: &mut R) {
                $(self.$idx.reset_params(rng));+
            }
        }

        /*This macro expands like this for a 4-tuple:

        impl<
            Input: Tensor,

            // `$last:`
            D:

            // `$(Module::<$rev_tail ::Output>, $rev_tail: )+`
            Module<C ::Output>, C:
            Module<B ::Output>, B:
            Module<A ::Output>, A:

            Module<Input>
        > Module<Input> for (A, B, C, D) {
            type Output = D::Output;
            fn forward(&self, x: Input) -> Self::Output {
                let x = self.0.forward(x);
                let x = self.1.forward(x);
                let x = self.2.forward(x);
                let x = self.3.forward(x);
                x
            }
        }
        */
        impl<
            Input: Tensor,
            $last:
            $(Module::<$rev_tail ::Output>, $rev_tail: )+
            Module<Input>
        > Module<Input> for ($($name,)+) {
            type Output = $last ::Output;

            /// Calls forward sequentially on each module in the tuple.
            fn forward(&self, x: Input) -> Self::Output {
                $(let x = self.$idx.forward(x);)+
                x
            }
        }

        impl<
            Input: Tensor,
            $last:
            $(ModuleMut::<$rev_tail ::Output>, $rev_tail: )+
            ModuleMut<Input>
        > ModuleMut<Input> for ($($name,)+) {
            type Output = $last ::Output;

            /// Calls forward sequentially on each module in the tuple.
            fn forward_mut(&mut self, x: Input) -> Self::Output {
                $(let x = self.$idx.forward_mut(x);)+
                x
            }
        }
    };
}

tuple_impls!([A, B] [0, 1], B, [A]);
tuple_impls!([A, B, C] [0, 1, 2], C, [B, A]);
tuple_impls!([A, B, C, D] [0, 1, 2, 3], D, [C, B, A]);
tuple_impls!([A, B, C, D, E] [0, 1, 2, 3, 4], E, [D, C, B, A]);
tuple_impls!([A, B, C, D, E, F] [0, 1, 2, 3, 4, 5], F, [E, D, C, B, A]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::tests::SimpleGradients;
    use crate::unique_id::HasUniqueId;
    use rand::{prelude::StdRng, SeedableRng};

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
            .forward_mut(Tensor1D::randn(&mut rng).traced())
            .square()
            .mean();
        let gradients = backward(loss);

        assert!(gradients.ref_gradient(&model.0.weight) != &[[0.0; 2]; 3]);
        assert!(gradients.ref_gradient(&model.0.bias) != &[0.0; 3]);
        assert!(gradients.ref_gradient(&model.1.weight) != &[[0.0; 3]; 4]);
        assert!(gradients.ref_gradient(&model.1.bias) != &[0.0; 4]);

        let mut sgd = Sgd::new(SgdConfig {
            lr: 1.0,
            momentum: None,
        });
        sgd.update(&mut model, gradients).expect("");

        assert!(model.0.weight.data() != m0.0.weight.data());
        assert!(model.0.bias.data() != m0.0.bias.data());
        assert!(model.1.weight.data() != m0.1.weight.data());
        assert!(model.1.bias.data() != m0.1.bias.data());
    }

    /// A struct to test the forward method of tuples. This sets the `I`th valuein a 1d tensors of size `N` to 1.0.
    #[derive(Debug, Default, Clone)]
    struct SetTo1<const I: usize, const N: usize>;

    impl<const I: usize, const N: usize> CanUpdateWithGradients for SetTo1<I, N> {
        fn update<G: GradientProvider>(&mut self, _: &mut G, _: &mut UnusedTensors) {}
    }
    impl<const I: usize, const N: usize> ResetParams for SetTo1<I, N> {
        fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {}
    }
    impl<const I: usize, const N: usize> Module<Tensor1D<N>> for SetTo1<I, N> {
        type Output = Tensor1D<N>;
        fn forward(&self, mut input: Tensor1D<N>) -> Self::Output {
            input.mut_data()[I] = 1.0;
            input
        }
    }

    #[test]
    fn test_set_to_1() {
        assert_eq!(
            SetTo1::<0, 5>::default().forward(Tensor1D::zeros()).data(),
            &[1.0, 0.0, 0.0, 0.0, 0.0]
        );

        assert_eq!(
            SetTo1::<1, 5>::default().forward(Tensor1D::zeros()).data(),
            &[0.0, 1.0, 0.0, 0.0, 0.0]
        );

        assert_eq!(
            SetTo1::<2, 5>::default().forward(Tensor1D::zeros()).data(),
            &[0.0, 0.0, 1.0, 0.0, 0.0]
        );

        assert_eq!(
            SetTo1::<3, 5>::default().forward(Tensor1D::zeros()).data(),
            &[0.0, 0.0, 0.0, 1.0, 0.0]
        );

        assert_eq!(
            SetTo1::<4, 5>::default().forward(Tensor1D::zeros()).data(),
            &[0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_2_tuple_forward() {
        let model: (SetTo1<0, 2>, SetTo1<1, 2>) = Default::default();
        let y = model.forward(Tensor1D::zeros());
        assert_eq!(y.data(), &[1.0, 1.0]);
    }

    #[test]
    fn test_3_tuple_forward() {
        let model: (SetTo1<0, 3>, SetTo1<1, 3>, SetTo1<2, 3>) = Default::default();
        let y = model.forward(Tensor1D::zeros());
        assert_eq!(y.data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_4_tuple_forward() {
        let model: (SetTo1<0, 4>, SetTo1<1, 4>, SetTo1<2, 4>, SetTo1<3, 4>) = Default::default();
        let y = model.forward(Tensor1D::zeros());
        assert_eq!(y.data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_5_tuple_forward() {
        let model: (
            SetTo1<0, 5>,
            SetTo1<1, 5>,
            SetTo1<2, 5>,
            SetTo1<3, 5>,
            SetTo1<4, 5>,
        ) = Default::default();
        let y = model.forward(Tensor1D::zeros());
        assert_eq!(y.data(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_6_tuple_forward() {
        let model: (
            SetTo1<0, 6>,
            SetTo1<1, 6>,
            SetTo1<2, 6>,
            SetTo1<3, 6>,
            SetTo1<4, 6>,
            SetTo1<5, 6>,
        ) = Default::default();
        let y = model.forward(Tensor1D::zeros());
        assert_eq!(y.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_tuple_missing_gradients() {
        let mut model: (Linear<5, 3>, Linear<5, 3>, Linear<5, 3>) = Default::default();
        let mut g: SimpleGradients = Default::default();

        // no gradients present
        let mut unused: UnusedTensors = Default::default();
        model.update(&mut g, &mut unused);
        assert_eq!(
            &unused.ids,
            &[
                *model.0.weight.id(),
                *model.0.bias.id(),
                *model.1.weight.id(),
                *model.1.bias.id(),
                *model.2.weight.id(),
                *model.2.bias.id(),
            ]
        );

        // weight gradient is present
        g.0.mut_gradient(&model.0.weight);
        g.0.mut_gradient(&model.1.weight);
        g.0.mut_gradient(&model.2.weight);

        let mut unused: UnusedTensors = Default::default();
        model.update(&mut g, &mut unused);
        assert_eq!(
            &unused.ids,
            &[*model.0.bias.id(), *model.1.bias.id(), *model.2.bias.id(),]
        );

        g.0.mut_gradient(&model.0.weight);
        g.0.mut_gradient(&model.0.bias);
        g.0.mut_gradient(&model.1.weight);
        g.0.mut_gradient(&model.1.bias);
        g.0.mut_gradient(&model.2.weight);
        g.0.mut_gradient(&model.2.bias);

        let mut unused: UnusedTensors = Default::default();
        model.update(&mut g, &mut unused);
        assert!(unused.is_empty());
    }
}
