use super::traits::Module;
use crate::gradients::GradientTape;
use crate::tensor::{OnGradientTape, Randomize, Tensor};
use rand::prelude::{Distribution, Rng};

impl<Input, A, B> Module<Input> for (A, B)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
{
    type Output = B::Output;
    fn forward(&mut self, x: &mut Input) -> Self::Output {
        let mut x = self.0.forward(x);
        self.1.forward(&mut x)
    }
}

impl<Input, A, B, C> Module<Input> for (A, B, C)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
    C: Module<B::Output>,
{
    type Output = C::Output;
    fn forward(&mut self, x: &mut Input) -> Self::Output {
        let mut x = self.0.forward(x);
        let mut x = self.1.forward(&mut x);
        self.2.forward(&mut x)
    }
}

impl<Input, A, B, C, D> Module<Input> for (A, B, C, D)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
    C: Module<B::Output>,
    D: Module<C::Output>,
{
    type Output = D::Output;
    fn forward(&mut self, x: &mut Input) -> Self::Output {
        let mut x = self.0.forward(x);
        let mut x = self.1.forward(&mut x);
        let mut x = self.2.forward(&mut x);
        self.3.forward(&mut x)
    }
}

impl<Input, A, B, C, D, E> Module<Input> for (A, B, C, D, E)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
    C: Module<B::Output>,
    D: Module<C::Output>,
    E: Module<D::Output>,
{
    type Output = E::Output;
    fn forward(&mut self, x: &mut Input) -> Self::Output {
        let mut x = self.0.forward(x);
        let mut x = self.1.forward(&mut x);
        let mut x = self.2.forward(&mut x);
        let mut x = self.3.forward(&mut x);
        self.4.forward(&mut x)
    }
}

impl<Input, A, B, C, D, E, F> Module<Input> for (A, B, C, D, E, F)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
    C: Module<B::Output>,
    D: Module<C::Output>,
    E: Module<D::Output>,
    F: Module<E::Output>,
{
    type Output = F::Output;
    fn forward(&mut self, x: &mut Input) -> Self::Output {
        let mut x = self.0.forward(x);
        let mut x = self.1.forward(&mut x);
        let mut x = self.2.forward(&mut x);
        let mut x = self.3.forward(&mut x);
        let mut x = self.4.forward(&mut x);
        self.5.forward(&mut x)
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] [$($idx:tt),+]) => {
        impl<$($name: OnGradientTape),+> OnGradientTape for ($($name,)+)
        {
            fn put_on(&mut self, tape: &mut GradientTape) {
                $(self.$idx.put_on(tape));+
            }
            fn update_with(&mut self, tape: &GradientTape) {
                $(self.$idx.update_with(tape));+
            }
        }

        impl<$($name: Randomize),+> Randomize for ($($name,)+)
        {
            fn randomize<R: Rng, DIST: Distribution<f32>>(&mut self, rng: &mut R, dist: &DIST) {
                $(self.$idx.randomize(rng, dist));+
            }
        }
    };
}

tuple_impls!([A, B] [0, 1]);
tuple_impls!([A, B, C] [0, 1, 2]);
tuple_impls!([A, B, C, D] [0, 1, 2, 3]);
tuple_impls!([A, B, C, D, E] [0, 1, 2, 3, 4]);
tuple_impls!([A, B, C, D, E, F] [0, 1, 2, 3, 4, 5]);
