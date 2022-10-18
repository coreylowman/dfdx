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

impl<Input: Tensor, Mod: Module<Input>> Module<(Input, )> for AddInto<(Mod, )>
{
    type Output = Mod::Output;
    
    fn forward(&self, x: (Input, )) -> Self::Output {
        let (head, ) = x;
        
        // no need for adding
        self.0.0.forward(head)
    
    }
}

impl<Input: Tensor, Mod: ModuleMut<Input>> ModuleMut<(Input, )> for AddInto<(Mod, )> {
    type Output = Mod::Output;
    
    fn forward_mut(&mut self, x: (Input, )) -> Self::Output {
        let (head, ) = x;
        self.0.0.forward_mut(head)
    }
}

// so what we want to do:
// split off the head from the tuple
// put the tail into an AddInto
// forward head and tail
// add them together
// return
macro_rules! tuple_impls {
    ($headin: ident $head:ident [$($tailsin:ident $tails:ident),+]) => {
        impl<
            $headin: Tensor,
            $(tailsin: Tensor,)+
            $head: Module<$headin>,
            $($tails: Module<$tailsin, Output = $head::Output>,)+
        > Module<($headin, $($tailsin,)+)> for AddInto<($head, $($tails,)+)> {
            type Output = $head::Output;
            
            fn forward(&self, x: ($headin, $($tailsin,)+)) -> Self::Output {
                let ($head, $($tails,)+) = x;
                let head = $head;
                let tails = $($tails)+;
                let tail = AddInto(tails);
                let (base, tape) = head.forward();
                
            }
        }
    }
} 
