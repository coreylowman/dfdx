use crate::{shapes::Dtype, tensor::*};

use super::{tensor_collection::*, BuildModule, BuildOnDevice, Module, ModuleMut, ToDevice};

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

impl<T: BuildOnDevice<D, E>, D: DeviceStorage, E: Dtype> BuildOnDevice<D, E> for AddInto<T> {
    type Built = AddInto<T::Built>;
}

impl<T: BuildModule<D, E>, D: DeviceStorage, E: Dtype> BuildModule<D, E> for AddInto<T> {
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self(BuildModule::try_build(device)?))
    }
}

impl<E: Dtype, D: DeviceStorage, T: TensorCollection<E, D>> TensorCollection<E, D> for AddInto<T> {
    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_module("0", |s| &s.0, |s| &mut s.0)
    }
}

impl<T: ToDevice<D>, D> ToDevice<D> for AddInto<T> {
    type Output = AddInto<T::Output>;
    fn to_device(&self, device: &D) -> Self::Output {
        AddInto(self.0.to_device(device))
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
        gradients::OwnedTape,
        nn::{builders::*, DeviceBuildExt},
        shapes::*,
        tests::{TestDevice, TestDtype},
    };

    type TestAddIntoCpu = AddInto<(Linear<2, 5>, Linear<3, 5>)>;
    #[allow(unused)]
    type TestAddInto<D> = OnDevice<TestAddIntoCpu, D>;

    #[test]
    fn test_add_into_2() {
        let dev: TestDevice = Default::default();
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
        ));
    }

    #[test]
    fn test_add_into_3() {
        let dev: TestDevice = Default::default();
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>, Linear<4, 5>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
            dev.zeros::<Rank1<4>>().traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
            dev.zeros::<Rank2<3, 4>>().traced(),
        ));
    }

    #[test]
    fn test_add_into_4() {
        let dev: TestDevice = Default::default();
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>, Linear<4, 5>, Linear<5, 5>)>;
        let m = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
            dev.zeros::<Rank1<4>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
            dev.zeros::<Rank2<3, 4>>().traced(),
            dev.zeros::<Rank2<3, 5>>().traced(),
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
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
            dev.zeros::<Rank1<4>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
            dev.zeros::<Rank1<6>>().traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
            dev.zeros::<Rank2<3, 4>>().traced(),
            dev.zeros::<Rank2<3, 5>>().traced(),
            dev.zeros::<Rank2<3, 6>>().traced(),
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
        let _: Tensor<Rank1<5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
            dev.zeros::<Rank1<4>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
            dev.zeros::<Rank1<6>>().traced(),
            dev.zeros::<Rank1<7>>().traced(),
        ));
        let _: Tensor<Rank2<3, 5>, _, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
            dev.zeros::<Rank2<3, 4>>().traced(),
            dev.zeros::<Rank2<3, 5>>().traced(),
            dev.zeros::<Rank2<3, 6>>().traced(),
            dev.zeros::<Rank2<3, 7>>().traced(),
        ));
    }

    #[test]
    fn longer_network() {
        let dev: TestDevice = Default::default();
        // check if it works in a longer neural net
        type Model = (AddInto<(Linear<5, 3>, Linear<5, 3>)>, ReLU, Linear<3, 1>);
        let mut model = dev.build_module::<Model, TestDtype>();
        let _: Tensor<Rank1<1>, _, _, OwnedTape<_>> = model.forward((
            dev.zeros::<Rank1<5>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
        ));
        let _: Tensor<Rank2<5, 1>, _, _, OwnedTape<_>> = model.forward((
            dev.zeros::<Rank2<5, 5>>().traced(),
            dev.zeros::<Rank2<5, 5>>().traced(),
        ));
        let _: Tensor<Rank1<1>, _, _, OwnedTape<_>> = model.forward_mut((
            dev.zeros::<Rank1<5>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
        ));
        let _: Tensor<Rank2<5, 1>, _, _, OwnedTape<_>> = model.forward_mut((
            dev.zeros::<Rank2<5, 5>>().traced(),
            dev.zeros::<Rank2<5, 5>>().traced(),
        ));
    }
}
