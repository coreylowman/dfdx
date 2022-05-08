use crate::prelude::CanUpdateWithGradients;

pub trait Module<Input>: Default + CanUpdateWithGradients {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}
