use crate::prelude::*;

/// Forwards the input to the output.
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct Id;

impl<Input> Module<Input> for Id {
    type Output = Input;
    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        Ok(x)
    }
}

pub type Id1 = (Id,);
pub type Id2 = (Id, Id);
pub type Id3 = (Id, Id, Id);
pub type Id4 = (Id, Id, Id, Id);
pub type Id5 = (Id, Id, Id, Id, Id);
pub type Id6 = (Id, Id, Id, Id, Id, Id);
