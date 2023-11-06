use crate::prelude::*;

/// Calls [crate::tensor_ops::add()]
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Add;

// TODO: macro for more tuples
// TODO: lower the requirement, as long as one of the values can be broadcast into the other one
// TODO: check if this works for constants

impl<Input> Module<(Input, Input)> for Add
where
    Input: TryAdd,
{
    type Output = <Input as TryAdd>::Output;

    fn try_forward(&self, x: (Input, Input)) -> Result<Self::Output, Error> {
        x.0.try_add(x.1)
    }
}
