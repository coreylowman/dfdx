use crate::{
    module_collection,
    traits::{Batch, Module, Tensor},
};

#[derive(Default, Debug)]
pub struct ModuleChain<M1: Module, M2: Module> {
    first: M1,
    second: M2,
}

module_collection!(
    [
        T: Tensor + Batch,
        M1: Module<Output = T>,
        M2: Module<Input = T>
    ],
    [M1, M2],
    ModuleChain,
    [first, second,]
);

impl<T: Tensor + Batch, M1: Module<Output = T>, M2: Module<Input = T>> Module
    for ModuleChain<M1, M2>
{
    type Input = M1::Input;
    type Output = M2::Output;

    fn forward<const B: usize>(
        &mut self,
        input: &mut <Self::Input as Batch>::Batched<B>,
    ) -> <Self::Output as Batch>::Batched<B> {
        let mut middle = self.first.forward(input);
        self.second.forward(&mut middle)
    }
}
