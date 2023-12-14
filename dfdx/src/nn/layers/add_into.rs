use crate::prelude::*;

/// Add inputs together into a single tensor. `T` should be a tuple
/// where every element of the tuple has the same output type
///
/// This provides a utility for networks where multiple inputs are needed
///
/// Generics:
/// - `T` the module to add the outputs together of
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = AddInto<(LinearConstConfig<2, 5>, LinearConstConfig<3, 5>)>;
/// let model = dev.build_module::<f32>(Model::default());
/// let a: Tensor<Rank1<2>, f32, _> = dev.zeros();
/// let b: Tensor<Rank1<3>, f32, _> = dev.zeros();
/// let _: Tensor<Rank1<5>, f32, _> = model.forward((a, b));
/// ```
#[derive(Debug, Default, Clone, ResetParams, ZeroGrads, WithGrads, UpdateParams)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
#[repr(transparent)]
pub struct AddInto<T>(
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub T,
);

impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>> BuildOnDevice<E, D> for AddInto<T> {
    type Built = AddInto<T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        let t = self.0.try_build_on_device(device)?;
        Ok(AddInto(t))
    }
}

macro_rules! sum {
    ($H:tt) => { $H };
    ($H:tt, $($T:tt),+) => { $H.try_add(sum!($($T),+))? };
}

macro_rules! add_into_impls {
    ($([$Mod:tt $ModVar:tt $Inp:tt $InpVar:tt]),+) => {
        impl<
            Out: TryAdd<Out, Output = Out>,
            Ai, $($Inp, )+
            A: Module<Ai, Output = Out>,
            $($Mod: Module<$Inp, Output = Out>, )+
        > Module<(Ai, $($Inp, )+)> for AddInto<(A, $($Mod, )+)>
        {
            type Output = Out;

            #[allow(clippy::needless_question_mark)]
            fn try_forward(&self, x: (Ai, $($Inp, )+)) -> Result<Self::Output, Error> {
                let (a, $($ModVar, )+) = &self.0;
                let (a_i, $($InpVar, )+) = x;
                let a_i = a.try_forward(a_i)?;
                $(let $InpVar = $ModVar.try_forward($InpVar)?;)+
                Ok(sum!(a_i, $($InpVar),*))
            }
            #[allow(clippy::needless_question_mark)]
            fn try_forward_mut(&mut self, x: (Ai, $($Inp, )+)) -> Result<Self::Output, Error> {
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
    use crate::tests::*;

    #[test]
    fn longer_network() {
        let dev: TestDevice = Default::default();
        // check if it works in a longer neural net
        type Model = (
            AddInto<(LinearConstConfig<5, 3>, LinearConstConfig<5, 3>)>,
            ReLU,
            LinearConstConfig<3, 1>,
        );
        let mut model = dev.build_module::<TestDtype>(Model::default());
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
