use crate::prelude::*;

/// Forwards the input to the output.
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct Id;

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Id {
    type Output = Tensor<S, E, D, T>;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        Ok(x)
    }
}

pub type Id1 = (Id,);
pub type Id2 = (Id, Id);
pub type Id3 = (Id, Id, Id);
pub type Id4 = (Id, Id, Id, Id);
pub type Id5 = (Id, Id, Id, Id, Id);
pub type Id6 = (Id, Id, Id, Id, Id, Id);
