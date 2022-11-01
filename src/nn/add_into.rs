use crate::prelude::*;
use dfdx_macros::CanUpdateWithGradients;

/// Add inputs together into a single tensor. `T` should be a tuple
/// where every element of the tuple has the same output type
///
/// This provides a utility for networks where multiple inputs are needed
///
/// # Generics
/// - `T` the module to add the outputs together of
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// type Model = AddInto<(Linear<2, 5>, Linear<3, 5>)>;
/// let model: Model = Default::default();
/// let _: Tensor1D<5> = model.forward((Tensor1D::<2>::zeros(), Tensor1D::<3>::zeros()));
/// ```
#[derive(Debug, Default, Clone, CanUpdateWithGradients)]
pub struct AddInto<T>(pub T);

impl<T: ResetParams> ResetParams for AddInto<T> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.0.reset_params(rng);
    }
}

macro_rules! tuple_impls {
    ($head:ident $headin:ident [$($tails:ident $tailsin:ident),+]) => {
        impl<
            Output: Tensor<Dtype = f32>,
            $headin: Tensor<Dtype = f32>,
            $($tailsin: Tensor<Dtype = f32>,)+
            $head: Module<$headin, Output = Output>,
            $($tails: Module<$tailsin, Output = Output>,)+
        > Module<($headin, $($tailsin,)+)> for AddInto<($head, $($tails,)+)> {
            type Output = Output;

            #[allow(non_snake_case)]
            fn forward(&self, x: ($headin, $($tailsin,)+)) -> Self::Output {

                // inputs
                let ($head, $($tails),+) = x;

                // layers
                let ($headin, $($tailsin),+) = &self.0;

                // forward
                let ($head, $($tails),+) = ($headin.forward($head), $($tailsin.forward($tails)),+);

                // add together
                $(
                    let $head = add($head, $tails);
                )+

                $head
            }
        }


        impl<
            Output: Tensor<Dtype = f32>,
            $headin: Tensor<Dtype = f32>,
            $($tailsin: Tensor<Dtype = f32>,)+
            $head: ModuleMut<$headin, Output = Output>,
            $($tails: ModuleMut<$tailsin, Output = Output>,)+
        > ModuleMut<($headin, $($tailsin,)+)> for AddInto<($head, $($tails,)+)> {
            type Output = Output;

            #[allow(non_snake_case)]
            fn forward_mut(&mut self, x: ($headin, $($tailsin,)+)) -> Self::Output {

                // inputs
                let ($head, $($tails),+) = x;

                // layers
                let ($headin, $($tailsin),+) = &mut self.0;

                // forward
                let ($head, $($tails),+) = ($headin.forward_mut($head), $($tailsin.forward_mut($tails)),+);

                // add together
                $(
                    let $head = add($head, $tails);
                )+

                $head
            }
        }    }
}

tuple_impls!(A Ai [B Bi]);
tuple_impls!(A Ai [B Bi, C Ci]);
tuple_impls!(A Ai [B Bi, C Ci, D Di]);
tuple_impls!(A Ai [B Bi, C Ci, D Di, E Ei]);
tuple_impls!(A Ai [B Bi, C Ci, D Di, E Ei, F Fi]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::tests::SimpleGradients, unique_id::HasUniqueId};

    #[test]
    fn test_add_into_2() {
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>)>;
        let m: Model = Default::default();
        let _: Tensor1D<5, OwnedTape> =
            m.forward((Tensor1D::zeros().traced(), Tensor1D::zeros().traced()));
        let _: Tensor2D<3, 5, OwnedTape> = m.forward((
            Tensor2D::<3, 2>::zeros().traced(),
            Tensor2D::<3, 3>::zeros().traced(),
        ));
    }

    #[test]
    fn test_add_into_3() {
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>, Linear<4, 5>)>;
        let m: Model = Default::default();
        let _: Tensor1D<5, OwnedTape> = m.forward((
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
        ));
        let _: Tensor2D<3, 5, OwnedTape> = m.forward((
            Tensor2D::<3, 2>::zeros().traced(),
            Tensor2D::<3, 3>::zeros().traced(),
            Tensor2D::<3, 4>::zeros().traced(),
        ));
    }

    #[test]
    fn test_add_into_4() {
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>, Linear<4, 5>, Linear<5, 5>)>;
        let m: Model = Default::default();
        let _: Tensor1D<5, OwnedTape> = m.forward((
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
        ));
        let _: Tensor2D<3, 5, OwnedTape> = m.forward((
            Tensor2D::<3, 2>::zeros().traced(),
            Tensor2D::<3, 3>::zeros().traced(),
            Tensor2D::<3, 4>::zeros().traced(),
            Tensor2D::<3, 5>::zeros().traced(),
        ));
    }

    #[test]
    fn test_add_into_5() {
        type Model = AddInto<(
            Linear<2, 5>,
            Linear<3, 5>,
            Linear<4, 5>,
            Linear<5, 5>,
            Linear<6, 5>,
        )>;
        let m: Model = Default::default();
        let _: Tensor1D<5, OwnedTape> = m.forward((
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
        ));
        let _: Tensor2D<3, 5, OwnedTape> = m.forward((
            Tensor2D::<3, 2>::zeros().traced(),
            Tensor2D::<3, 3>::zeros().traced(),
            Tensor2D::<3, 4>::zeros().traced(),
            Tensor2D::<3, 5>::zeros().traced(),
            Tensor2D::<3, 6>::zeros().traced(),
        ));
    }

    #[test]
    fn test_add_into_6() {
        type Model = AddInto<(
            Linear<2, 5>,
            Linear<3, 5>,
            Linear<4, 5>,
            Linear<5, 5>,
            Linear<6, 5>,
            Linear<7, 5>,
        )>;
        let m: Model = Default::default();
        let _: Tensor1D<5, OwnedTape> = m.forward((
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
            Tensor1D::zeros().traced(),
        ));
        let _: Tensor2D<3, 5, OwnedTape> = m.forward((
            Tensor2D::<3, 2>::zeros().traced(),
            Tensor2D::<3, 3>::zeros().traced(),
            Tensor2D::<3, 4>::zeros().traced(),
            Tensor2D::<3, 5>::zeros().traced(),
            Tensor2D::<3, 6>::zeros().traced(),
            Tensor2D::<3, 7>::zeros().traced(),
        ));
    }

    #[test]
    fn test_missing_gradients() {
        let mut model: AddInto<(Linear<5, 3>, Linear<5, 3>)> = Default::default();
        let mut g: SimpleGradients = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused);
        assert_eq!(
            &unused.ids,
            &[
                *model.0 .0.weight.id(),
                *model.0 .0.bias.id(),
                *model.0 .1.weight.id(),
                *model.0 .1.bias.id()
            ]
        );

        // weight gradient is present
        g.0.mut_gradient(&model.0 .0.weight);
        g.0.mut_gradient(&model.0 .0.bias);
        g.0.mut_gradient(&model.0 .1.weight);
        g.0.mut_gradient(&model.0 .1.bias);

        let mut unused = Default::default();
        model.update(&mut g, &mut unused);
        assert!(unused.is_empty());
    }
}
