use super::module::{Init, Module};
use crate::gradients::{GradientTape, Taped};
use crate::prelude::Tensor;
use ndarray_rand::rand::Rng;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct ModuleChain<A, B, INNER> {
    a: A,
    b: B,
    marker: PhantomData<INNER>,
}

impl<A, B, INNER> Default for ModuleChain<A, B, INNER>
where
    A: Default,
    B: Default,
{
    fn default() -> Self {
        Self {
            a: Default::default(),
            b: Default::default(),
            marker: PhantomData,
        }
    }
}

impl<A, B, INNER> Init for ModuleChain<A, B, INNER>
where
    A: Init,
    B: Init,
{
    fn init<R: Rng>(&mut self, rng: &mut R) {
        self.a.init(rng);
        self.b.init(rng);
    }
}

impl<A, B, INNER> Taped for ModuleChain<A, B, INNER>
where
    A: Taped,
    B: Taped,
{
    fn update(&mut self, tape: &GradientTape) {
        self.a.update(tape);
        self.b.update(tape);
    }
}

impl<A, B, I, INNER, O> Module<I, O> for ModuleChain<A, B, INNER>
where
    I: Tensor,
    INNER: Tensor,
    O: Tensor,
    A: Module<I, INNER>,
    B: Module<INNER, O>,
{
    fn forward(&mut self, input: &mut I) -> O {
        self.b.forward(&mut self.a.forward(input))
    }
}

pub trait Chain<I, INNER, O>: Sized {
    fn chain<B>(self) -> ModuleChain<Self, B, INNER>
    where
        INNER: Tensor,
        O: Tensor,
        B: Module<INNER, O>;
}

impl<T, I, INNER, O> Chain<I, INNER, O> for T
where
    I: Tensor,
    INNER: Tensor,
    T: Module<I, INNER>,
{
    fn chain<B>(self) -> ModuleChain<Self, B, INNER>
    where
        INNER: Tensor,
        O: Tensor,
        B: Module<INNER, O>,
    {
        ModuleChain {
            a: self,
            b: Default::default(),
            marker: PhantomData,
        }
    }
}
