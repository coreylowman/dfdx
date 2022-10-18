use crate::gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors};
use crate::prelude::*;

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
/// type Model = AddInto<(Linear<2, 5>, Linear<3, 5>)>;
/// let model: Model = Default::default();
/// let _: Tensor1D<5> = model.forward((Tensor1D::<2>::zeros(), Teensor1D::<3>::zeros()));
/// ```
#[derive(Debug, Default, Clone)]
pub struct AddInto<T>(pub T);

impl<T: CanUpdateWithGradients> CanUpdateWithGradients for AddInto<T> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.0.update(grads, unused);
    }
}

impl<T: ResetParams> ResetParams for AddInto<T> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.0.reset_params(rng);
    }
}

// so what we want to do:
// split off the head from the tuple
// put the tail into an AddInto
// forward head and tail
// add them together
// return
macro_rules! tuple_impls {
    ($head:ident $[$tails:ident,+]) => {
        impl<
            Output: Tensor,
            $head: Module<Input>,
            $($tails: Module<Input, Output = $head::Output>,)+
        > Module<Input> for AddInto<($head, $($tails,)+)> {
            type Output = $head::Output;
            
            fn forward(&self, x: T) -> Self::Output {
                
            }
        }
    }
} 
