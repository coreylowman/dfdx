use crate::{
    dtypes::Dtype,
    tensor::{Error, UniqueId},
    tensor_ops::Device,
};

use std::vec::Vec;

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+], $last:ident, [$($rev_tail:ident),*]) => {

        impl<Dev: Device<Elem>, Elem: Dtype, $($name: crate::nn_traits::BuildOnDevice<Elem, Dev>),+> crate::nn_traits::BuildOnDevice<Elem, Dev> for ($($name,)+) {
            type Built = ($($name::Built, )+);
            fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, Error> {
                Ok(($(
                    self.$idx.try_build_on_device(device)?,
                )+))
            }
        }

        #[cfg(feature = "safetensors")]
        impl<$($name: crate::nn_traits::SaveSafeTensors, )+> crate::nn_traits::SaveSafeTensors for ($($name,)+) {
            fn write_safetensors_with<F: FnMut(String) -> String>(
                &self,
                location: &str,
                tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
                key_map: &mut F,
            ) {
                $(
                    let name = &format!("{location}.{}", $idx);
                    self.$idx.write_safetensors_with(name, tensors, key_map);
                )+
            }
        }

        #[cfg(feature = "safetensors")]
        impl<$($name: crate::nn_traits::LoadSafeTensors, )+> crate::nn_traits::LoadSafeTensors for ($($name,)+) {
            fn read_safetensors_with<F: FnMut(String) -> String>(
                &mut self,
                location: &str,
                tensors: &safetensors::SafeTensors,
                skip_missing: bool,
                key_map: &mut F,
            ) -> Result<(), safetensors::SafeTensorError> {
                $(
                    let name = &format!("{location}.{}", $idx);
                    self.$idx.read_safetensors_with(name, tensors, skip_missing, key_map)?;
                )+
                Ok(())
            }
        }

        impl<Dev: Device<Elem>, Elem: Dtype, $($name: crate::nn_traits::ResetParams<Elem, Dev>),+> crate::nn_traits::ResetParams<Elem, Dev> for ($($name,)+) {
            fn try_reset_params(&mut self) -> Result<(), Error> {
                $(self.$idx.try_reset_params()?;)+
                Ok(())
            }
        }

        impl<Dev: Device<Elem>, Elem: Dtype, $($name: crate::nn_traits::UpdateParams<Elem, Dev>),+> crate::nn_traits::UpdateParams<Elem, Dev> for ($($name,)+) {
            fn try_update_params<M, Optim: crate::nn_traits::Optimizer<M, Elem, Dev>>(
                &mut self,
                optimizer: &mut Optim,
                gradients: &crate::prelude::Gradients<Elem, Dev>,
                missing_tensors: &mut Vec<UniqueId>,
            ) -> Result<(), Error> {
                $(self.$idx.try_update_params(optimizer, gradients, missing_tensors)?;)+
                Ok(())
            }
        }

        impl<Dev: Device<Elem>, Elem: Dtype, $($name: crate::nn_traits::ZeroGrads<Elem, Dev>),+> crate::nn_traits::ZeroGrads<Elem, Dev> for ($($name,)+) {
            fn try_zero_grads(&self, grads: &mut crate::prelude::Gradients<Elem, Dev>) -> Result<(), Error> {
                $(self.$idx.try_zero_grads(grads)?;)+
                Ok(())
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
            Input,
            $last:
            $(crate::nn_traits::Module::<$rev_tail ::Output>, $rev_tail: )*
            crate::nn_traits::Module<Input>
        > crate::nn_traits::Module<Input> for ($($name,)+) {
            type Output = $last ::Output;

            /// Calls forward sequentially on each module in the tuple.
            fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
                $(let x = self.$idx.try_forward(x)?;)+
                Ok(x)
            }

            /// Calls forward sequentially on each module in the tuple.
            fn try_forward_mut(&mut self, x: Input) -> Result<Self::Output, Error> {
                $(let x = self.$idx.try_forward_mut(x)?;)+
                Ok(x)
            }
        }
    };
}

tuple_impls!([M1][0], M1, []);
tuple_impls!([M1, M2] [0, 1], M2, [M1]);
tuple_impls!([M1, M2, M3] [0, 1, 2], M3, [M2, M1]);
tuple_impls!([M1, M2, M3, M4] [0, 1, 2, 3], M4, [M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5] [0, 1, 2, 3, 4], M5, [M4, M3, M2, M1]);
tuple_impls!([M1, M2, M3, M4, M5, M6] [0, 1, 2, 3, 4, 5], M6, [M5, M4, M3, M2, M1]);
