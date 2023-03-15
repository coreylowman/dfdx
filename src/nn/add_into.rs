use crate::{prelude::Device, shapes::Dtype};

use super::*;

/// Add inputs together into a single tensor. `T` should be a tuple
//// where every element of the tuple has the same output type
///
/// This provides a utility for networks where multiple inputs are needed
///
/// # Generics
/// - `T` the module to add the outputs together of
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// type Model = AddInto<(Linear<2, 5>, Linear<3, 5>)>;
/// let model = dev.build_module::<Model, f32>();
/// let a: Tensor<Rank1<2>, f32, _> = dev.zeros();
/// let b: Tensor<Rank1<3>, f32, _> = dev.zeros();
/// let _: Tensor<Rank1<5>, f32, _> = model.forward((a, b));
/// ```
#[derive(Debug, Default, Clone)]
pub struct AddInto<T>(pub T);

impl<T: BuildOnDevice<D, E>, D: Device<E>, E: Dtype> BuildOnDevice<D, E> for AddInto<T> {
    type Built = AddInto<T::Built>;
}

impl<E: Dtype, D: Device<E>, T: TensorCollection<E, D>> TensorCollection<E, D> for AddInto<T> {
    type To<E2: Dtype, D2: Device<E2>> = AddInto<T::To<E2, D2>>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(Self::module("0", |s| &s.0, |s| &mut s.0), AddInto)
    }
}

macro_rules! sum {
    ($H:tt) => { $H };
    ($H:tt, $($T:tt),+) => { $H + sum!($($T),+) };
}

macro_rules! add_into_impls {
    ($([$Mod:tt $ModVar:tt $Inp:tt $InpVar:tt]),+) => {
        impl<
            Out: std::ops::Add<Out, Output = Out>,
            Ai, $($Inp, )+
            A: Module<Ai, Output = Out>,
            $($Mod: Module<$Inp, Output = Out, Error = A::Error>, )+
        > Module<(Ai, $($Inp, )+)> for AddInto<(A, $($Mod, )+)>
        {
            type Output = Out;
            type Error = A::Error;

            fn try_forward(&self, x: (Ai, $($Inp, )+)) -> Result<Self::Output, Self::Error> {
                let (a, $($ModVar, )+) = &self.0;
                let (a_i, $($InpVar, )+) = x;
                let a_i = a.try_forward(a_i)?;
                $(let $InpVar = $ModVar.try_forward($InpVar)?;)+
                Ok(sum!(a_i, $($InpVar),*))
            }
        }
        impl<
            Out: std::ops::Add<Out, Output = Out>,
            Ai, $($Inp, )+
            A: ModuleMut<Ai, Output = Out>,
            $($Mod: ModuleMut<$Inp, Output = Out, Error = A::Error>, )+
        > ModuleMut<(Ai, $($Inp, )+)> for AddInto<(A, $($Mod, )+)>
        {
            type Output = Out;
            type Error = A::Error;

            fn try_forward_mut(&mut self, x: (Ai, $($Inp, )+)) -> Result<Self::Output, Self::Error> {
                let (a, $($ModVar, )+) = &mut self.0;
                let (a_i, $($InpVar, )+) = x;
                let a_i = a.try_forward_mut(a_i)?;
                $(let $InpVar = $ModVar.try_forward_mut($InpVar)?;)+
                Ok(sum!(a_i, $($InpVar),*))
            }
        }
    };
}

add_into_impls!([B b Bi b_i]);
add_into_impls!([B b Bi b_i], [C c Ci c_i]);
add_into_impls!([B b Bi b_i], [C c Ci c_i], [D d Di d_i]);
add_into_impls!([B b Bi b_i], [C c Ci c_i], [D d Di d_i], [E e Ei e_i]);
add_into_impls!([B b Bi b_i], [C c Ci c_i], [D d Di d_i], [E e Ei e_i], [F f Fi f_i]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::builders::*,
        prelude::{OwnedTape, Tensor, Trace, ZerosTensor},
        shapes::*,
        tests::{TestDevice, TestDtype},
    };

    #[test]
    fn test_add_into_2() {
        let dev: TestDevice = Default::default();
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank1<2>>().leaky_traced(),
            dev.zeros::<Rank1<3>>().leaky_traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().leaky_traced(),
            dev.zeros::<Rank2<3, 3>>().leaky_traced(),
        ));
    }

    #[test]
    fn test_add_into_3() {
        let dev: TestDevice = Default::default();
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>, Linear<4, 5>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank1<2>>().leaky_traced(),
            dev.zeros::<Rank1<3>>().leaky_traced(),
            dev.zeros::<Rank1<4>>().leaky_traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().leaky_traced(),
            dev.zeros::<Rank2<3, 3>>().leaky_traced(),
            dev.zeros::<Rank2<3, 4>>().leaky_traced(),
        ));
    }

    #[test]
    fn test_add_into_4() {
        let dev: TestDevice = Default::default();
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>, Linear<4, 5>, Linear<5, 5>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank1<2>>().leaky_traced(),
            dev.zeros::<Rank1<3>>().leaky_traced(),
            dev.zeros::<Rank1<4>>().leaky_traced(),
            dev.zeros::<Rank1<5>>().leaky_traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().leaky_traced(),
            dev.zeros::<Rank2<3, 3>>().leaky_traced(),
            dev.zeros::<Rank2<3, 4>>().leaky_traced(),
            dev.zeros::<Rank2<3, 5>>().leaky_traced(),
        ));
    }

    #[test]
    fn test_add_into_5() {
        let dev: TestDevice = Default::default();
        type Model = AddInto<(
            Linear<2, 5>,
            Linear<3, 5>,
            Linear<4, 5>,
            Linear<5, 5>,
            Linear<6, 5>,
        )>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank1<2>>().leaky_traced(),
            dev.zeros::<Rank1<3>>().leaky_traced(),
            dev.zeros::<Rank1<4>>().leaky_traced(),
            dev.zeros::<Rank1<5>>().leaky_traced(),
            dev.zeros::<Rank1<6>>().leaky_traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().leaky_traced(),
            dev.zeros::<Rank2<3, 3>>().leaky_traced(),
            dev.zeros::<Rank2<3, 4>>().leaky_traced(),
            dev.zeros::<Rank2<3, 5>>().leaky_traced(),
            dev.zeros::<Rank2<3, 6>>().leaky_traced(),
        ));
    }

    #[test]
    fn test_add_into_6() {
        let dev: TestDevice = Default::default();
        type Model = AddInto<(
            Linear<2, 5>,
            Linear<3, 5>,
            Linear<4, 5>,
            Linear<5, 5>,
            Linear<6, 5>,
            Linear<7, 5>,
        )>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank1<2>>().leaky_traced(),
            dev.zeros::<Rank1<3>>().leaky_traced(),
            dev.zeros::<Rank1<4>>().leaky_traced(),
            dev.zeros::<Rank1<5>>().leaky_traced(),
            dev.zeros::<Rank1<6>>().leaky_traced(),
            dev.zeros::<Rank1<7>>().leaky_traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_, _>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().leaky_traced(),
            dev.zeros::<Rank2<3, 3>>().leaky_traced(),
            dev.zeros::<Rank2<3, 4>>().leaky_traced(),
            dev.zeros::<Rank2<3, 5>>().leaky_traced(),
            dev.zeros::<Rank2<3, 6>>().leaky_traced(),
            dev.zeros::<Rank2<3, 7>>().leaky_traced(),
        ));
    }

    #[test]
    fn longer_network() {
        let dev: TestDevice = Default::default();
        // check if it works in a longer neural net
        type Model = (AddInto<(Linear<5, 3>, Linear<5, 3>)>, ReLU, Linear<3, 1>);
        let mut model = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<1>, _, _, OwnedTape<_, _>> = model.forward((
            dev.zeros::<Rank1<5>>().leaky_traced(),
            dev.zeros::<Rank1<5>>().leaky_traced(),
        ));
        let _: Tensor<Rank2<5, 1>, _, _, OwnedTape<_, _>> = model.forward((
            dev.zeros::<Rank2<5, 5>>().leaky_traced(),
            dev.zeros::<Rank2<5, 5>>().leaky_traced(),
        ));
        let _: Tensor<Rank1<1>, _, _, OwnedTape<_, _>> = model.forward_mut((
            dev.zeros::<Rank1<5>>().leaky_traced(),
            dev.zeros::<Rank1<5>>().leaky_traced(),
        ));
        let _: Tensor<Rank2<5, 1>, _, _, OwnedTape<_, _>> = model.forward_mut((
            dev.zeros::<Rank2<5, 5>>().leaky_traced(),
            dev.zeros::<Rank2<5, 5>>().leaky_traced(),
        ));
    }
}
