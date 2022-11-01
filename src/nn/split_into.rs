use crate::prelude::*;
use dfdx_macros::{CanUpdateWithGradients, ResetParams};
use rand::Rng;

/// Splits input into multiple heads. `T` should be a tuple,
/// where every element of the tuple accepts the same input type.
///
/// This provides a utility for multi headed structures where
/// the tape needs to be moved around a number of times.
///
/// # Generics
/// - `T` the module to split the input into.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// type Model = SplitInto<(Linear<5, 3>, Linear<5, 7>)>;
/// let model: Model = Default::default();
/// let _: (Tensor1D<3>, Tensor1D<7>) = model.forward(Tensor1D::<5>::zeros());
/// ```
#[derive(Debug, Default, Clone, CanUpdateWithGradients, ResetParams)]
pub struct SplitInto<T>(pub T);

macro_rules! tuple_impls {
    ([$($heads:ident),+] $tail:ident) => {
impl<
    Input: Tensor,
    $($heads : Module<Input>,)+
    $tail: Module<Input>
> Module<Input> for SplitInto<($($heads,)+ $tail)>
where
    $($heads::Output: Tensor<Tape = Input::Tape>,)+
{
    type Output = (
        $(<$heads::Output as Tensor>::NoTape, )+
        $tail::Output
    );

    #[allow(non_snake_case)]
    fn forward(&self, x: Input) -> Self::Output {
        let (x, tape) = x.split_tape();
        let ($($heads, )+ $tail) = &self.0;
        $(let ($heads, tape) = $heads.forward(x.clone().put_tape(tape)).split_tape();)+
        let $tail = $tail.forward(x.put_tape(tape));
        (
            $($heads,)+
            $tail
        )
    }
}

impl<
    Input: Tensor,
    $($heads : ModuleMut<Input>,)+
    $tail: ModuleMut<Input>
> ModuleMut<Input> for SplitInto<($($heads,)+ $tail)>
where
    $($heads::Output: Tensor<Tape = Input::Tape>,)+
{
    type Output = (
        $(<$heads::Output as Tensor>::NoTape, )+
        $tail::Output
    );

    #[allow(non_snake_case)]
    fn forward_mut(&mut self, x: Input) -> Self::Output {
        let (x, tape) = x.split_tape();
        let ($($heads, )+ $tail) = &mut self.0;
        $(let ($heads, tape) = $heads.forward_mut(x.clone().put_tape(tape)).split_tape();)+
        let $tail = $tail.forward_mut(x.put_tape(tape));
        (
            $($heads,)+
            $tail
        )
    }
}
}
}

tuple_impls!([A] B);
tuple_impls!([A, B] C);
tuple_impls!([A, B, C] D);
tuple_impls!([A, B, C, D] E);
tuple_impls!([A, B, C, D, E] F);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::tests::SimpleGradients, unique_id::HasUniqueId};

    #[test]
    fn test_split_into_2() {
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>)>;
        let m: Model = Default::default();
        let _: (Tensor1D<1>, Tensor1D<2, OwnedTape>) = m.forward(Tensor1D::zeros().traced());
        let _: (Tensor2D<3, 1>, Tensor2D<3, 2, OwnedTape>) =
            m.forward(Tensor2D::<3, 5>::zeros().traced());
    }

    #[test]
    fn test_split_into_3() {
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>)>;
        let m: Model = Default::default();
        let _: (Tensor1D<1>, Tensor1D<2>, Tensor1D<3, OwnedTape>) =
            m.forward(Tensor1D::zeros().traced());
        let _: (Tensor2D<3, 1>, Tensor2D<3, 2>, Tensor2D<3, 3, OwnedTape>) =
            m.forward(Tensor2D::<3, 5>::zeros().traced());
    }

    #[test]
    fn test_split_into_4() {
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>, Linear<5, 4>)>;
        let m: Model = Default::default();
        let _: (
            Tensor1D<1>,
            Tensor1D<2>,
            Tensor1D<3>,
            Tensor1D<4, OwnedTape>,
        ) = m.forward(Tensor1D::zeros().traced());
        let _: (
            Tensor2D<3, 1>,
            Tensor2D<3, 2>,
            Tensor2D<3, 3>,
            Tensor2D<3, 4, OwnedTape>,
        ) = m.forward(Tensor2D::<3, 5>::zeros().traced());
    }

    #[test]
    fn test_split_into_5() {
        type Model = SplitInto<(
            Linear<5, 1>,
            Linear<5, 2>,
            Linear<5, 3>,
            Linear<5, 4>,
            Linear<5, 5>,
        )>;
        let m: Model = Default::default();
        let _: (
            Tensor1D<1>,
            Tensor1D<2>,
            Tensor1D<3>,
            Tensor1D<4>,
            Tensor1D<5, OwnedTape>,
        ) = m.forward(Tensor1D::zeros().traced());
        let _: (
            Tensor2D<3, 1>,
            Tensor2D<3, 2>,
            Tensor2D<3, 3>,
            Tensor2D<3, 4>,
            Tensor2D<3, 5, OwnedTape>,
        ) = m.forward(Tensor2D::<3, 5>::zeros().traced());
    }

    #[test]
    fn test_split_into_6() {
        type Model = SplitInto<(
            Linear<5, 1>,
            Linear<5, 2>,
            Linear<5, 3>,
            Linear<5, 4>,
            Linear<5, 5>,
            Linear<5, 6>,
        )>;
        let m: Model = Default::default();
        let _: (
            Tensor1D<1>,
            Tensor1D<2>,
            Tensor1D<3>,
            Tensor1D<4>,
            Tensor1D<5>,
            Tensor1D<6, OwnedTape>,
        ) = m.forward(Tensor1D::zeros().traced());
        let _: (
            Tensor2D<3, 1>,
            Tensor2D<3, 2>,
            Tensor2D<3, 3>,
            Tensor2D<3, 4>,
            Tensor2D<3, 5>,
            Tensor2D<3, 6, OwnedTape>,
        ) = m.forward(Tensor2D::<3, 5>::zeros().traced());
    }

    #[test]
    fn test_missing_gradients() {
        let mut model: SplitInto<(Linear<5, 3>, Linear<5, 3>)> = Default::default();
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
