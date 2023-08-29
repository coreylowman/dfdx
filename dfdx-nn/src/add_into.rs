use crate::*;

use dfdx::{
    shapes::Dtype,
    tensor_ops::{Device, TryAdd},
};

/// Add inputs together into a single tensor. `T` should be a tuple
//// where every element of the tuple has the same output type
///
/// This provides a utility for networks where multiple inputs are needed
///
/// Generics:
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
#[derive(
    Debug, Default, Clone, ResetParams, ZeroGrads, UpdateParams, LoadSafeTensors, SaveSafeTensors,
)]
#[repr(transparent)]
pub struct AddInto<T>(
    #[module]
    #[serialize]
    pub T,
);

impl<E: Dtype, D: Device<E>, T: BuildOnDevice<E, D>> BuildOnDevice<E, D> for AddInto<T> {
    type Built = AddInto<T::Built>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, <D>::Err> {
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
            Out: TryAdd<Out, Output = Out, Err = A::Error>,
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
