use crate::prelude::*;

/// Splits input into multiple heads. `T` should be a tuple,
/// where every element of the tuple accepts the same input type.
///
/// This provides a utility for multi headed structures where
/// the tape needs to be moved around a number of times.
///
/// Implements:
/// - [Debug], [Default], [Clone]
/// - [Module<(A, B, ...)>]
/// - [ResetParams]
/// - [CanUpdateWithGradients]
/// - [SaveToNpz]
/// - [LoadFromNpz]
///
/// Generics:
/// - `T` the module to split the input into.
///
/// # Example:
/// ```rust
/// # use dfdx::prelude::*;
/// type Model = SplitInto<(Linear<5, 3>, Linear<5, 7>)>;
/// let model: Model = Default::default();
/// let _: (Tensor1D<3>, Tensor1D<7>) = model.forward(Tensor1D::<5>::zeros());
/// ```
#[derive(Debug, Default, Clone)]
pub struct SplitInto<T>(T);

impl<T: CanUpdateWithGradients> CanUpdateWithGradients for SplitInto<T> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.0.update(grads);
    }
}

impl<T: ResetParams> ResetParams for SplitInto<T> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.0.reset_params(rng);
    }
}

impl<T: SaveToNpz> SaveToNpz for SplitInto<T> {
    fn write<W>(&self, p: &str, w: &mut zip::ZipWriter<W>) -> zip::result::ZipResult<()>
    where
        W: std::io::Write + std::io::Seek,
    {
        self.0.write(p, w)
    }
}

impl<T: LoadFromNpz> LoadFromNpz for SplitInto<T> {
    fn read<R>(&mut self, p: &str, r: &mut zip::ZipArchive<R>) -> Result<(), NpzError>
    where
        R: std::io::Read + std::io::Seek,
    {
        self.0.read(p, r)
    }
}

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
        $(let ($heads, tape) = $heads.forward(x.duplicate().put_tape(tape)).split_tape();)+
        let $tail = $tail.forward(x.put_tape(tape));
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

    #[test]
    fn test_split_into_2() {
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>)>;
        let m: Model = Default::default();
        let _: (Tensor1D<1>, Tensor1D<2, OwnsTape>) = m.forward(Tensor1D::zeros().traced());
        let _: (Tensor2D<3, 1>, Tensor2D<3, 2, OwnsTape>) =
            m.forward(Tensor2D::<3, 5>::zeros().traced());
    }

    #[test]
    fn test_split_into_3() {
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>)>;
        let m: Model = Default::default();
        let _: (Tensor1D<1>, Tensor1D<2>, Tensor1D<3, OwnsTape>) =
            m.forward(Tensor1D::zeros().traced());
        let _: (Tensor2D<3, 1>, Tensor2D<3, 2>, Tensor2D<3, 3, OwnsTape>) =
            m.forward(Tensor2D::<3, 5>::zeros().traced());
    }

    #[test]
    fn test_split_into_4() {
        type Model = SplitInto<(Linear<5, 1>, Linear<5, 2>, Linear<5, 3>, Linear<5, 4>)>;
        let m: Model = Default::default();
        let _: (Tensor1D<1>, Tensor1D<2>, Tensor1D<3>, Tensor1D<4, OwnsTape>) =
            m.forward(Tensor1D::zeros().traced());
        let _: (
            Tensor2D<3, 1>,
            Tensor2D<3, 2>,
            Tensor2D<3, 3>,
            Tensor2D<3, 4, OwnsTape>,
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
            Tensor1D<5, OwnsTape>,
        ) = m.forward(Tensor1D::zeros().traced());
        let _: (
            Tensor2D<3, 1>,
            Tensor2D<3, 2>,
            Tensor2D<3, 3>,
            Tensor2D<3, 4>,
            Tensor2D<3, 5, OwnsTape>,
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
            Tensor1D<6, OwnsTape>,
        ) = m.forward(Tensor1D::zeros().traced());
        let _: (
            Tensor2D<3, 1>,
            Tensor2D<3, 2>,
            Tensor2D<3, 3>,
            Tensor2D<3, 4>,
            Tensor2D<3, 5>,
            Tensor2D<3, 6, OwnsTape>,
        ) = m.forward(Tensor2D::<3, 5>::zeros().traced());
    }
}
