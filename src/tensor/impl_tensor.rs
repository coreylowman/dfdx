use super::structs::*;
use super::traits::*;
use crate::diff_fns::*;
use crate::gradients::{Gradient, GradientTape, HasGradient, OnGradientTape};
use ndarray::{Array, Ix0, Ix1, Ix2, Ix3, Ix4};
use rand::prelude::{Distribution, Rng};
use rand_distr::{Standard, StandardNormal};

macro_rules! tensor_impl {
    ([$($const_defs:tt)*] $typename:ident [$($consts:tt)*], $dim:ty, $shape_val:expr, $shape_type:ty, $num_elems:expr) => {
        impl<$($const_defs)*> IsShapedArray for $typename<$($consts)*> {
            type Dimension = $dim;
            type Shape = $shape_type;
            const SHAPE: Self::Shape = $shape_val;
            const NUM_ELEMENTS: usize = $num_elems;

            fn data(&self) -> &Array<f32, Self::Dimension> { &self.data }
            fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> { &mut self.data }
        }

        impl<$($const_defs)*> Default for $typename<$($consts)*> {
            fn default() -> Self {
                Self {
                    data: Array::zeros(Self::SHAPE),
                    grad: None,
                }
            }
        }

        impl<$($const_defs)*> HasGradient for $typename<$($consts)*> {
            fn grad(&self) -> &Option<Gradient> { &self.grad }
            fn mut_grad(&mut self) -> &mut Option<Gradient> { &mut self.grad }
        }

        impl<$($const_defs)*> OnGradientTape for $typename<$($consts)*> {
            fn update(&mut self, tape: &GradientTape) {
                let grad = self.mut_grad().take().unwrap();
                *self.mut_data() -= &tape[grad.gradient_ref];
            }
        }

        impl<$($const_defs)*> Randomize for $typename<$($consts)*>
        {
            fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
                self.mut_data().map_inplace(|f| *f = dist.sample(rng))
            }
        }

        impl<$($const_defs)*> $typename<$($consts)*>
        {
            pub(super) fn record_on(&mut self, tape: &mut GradientTape) {
                if self.grad().is_none() {
                    *self.mut_grad() = Some(Gradient::new(tape.register_gradient(Self::SHAPE)));
                }
            }
        }

        impl<$($const_defs)*> Tensor for $typename<$($consts)*> { }
    }
}

tensor_impl!([] Tensor0D [], Ix0, (), (), 1);
tensor_impl!([const N: usize] Tensor1D [N], Ix1, (N,), (usize,), N);
tensor_impl!([const M: usize, const N: usize] Tensor2D [M, N], Ix2, (M, N), (usize, usize), M * N);
tensor_impl!([const M: usize, const N: usize, const O: usize] Tensor3D [M, N, O], Ix3, (M, N, O), (usize, usize, usize), M * N * O);
tensor_impl!([const M: usize, const N: usize, const O: usize, const P: usize] Tensor4D [M, N, O, P], Ix4, (M, N, O, P), (usize, usize, usize, usize), M * N * O * P);

impl<T> TensorSugar for T
where
    T: Tensor,
{
    fn zeros() -> Self {
        let mut a = Self::default();
        a.mut_data().fill(0.0);
        a
    }

    fn ones() -> Self {
        let mut a = Self::default();
        a.mut_data().fill(1.0);
        a
    }

    fn rand<R: Rng>(rng: &mut R) -> Self {
        let mut a = Self::default();
        a.mut_data().map_inplace(|f| *f = Standard.sample(rng));
        a
    }

    fn randn<R: Rng>(rng: &mut R) -> Self {
        let mut a = Self::default();
        a.mut_data()
            .map_inplace(|f| *f = StandardNormal.sample(rng));
        a
    }

    fn relu(&mut self) -> Self {
        self.apply::<ReLU>()
    }

    fn sin(&mut self) -> Self {
        self.apply::<Sin>()
    }

    fn cos(&mut self) -> Self {
        self.apply::<Cos>()
    }

    fn ln(&mut self) -> Self {
        self.apply::<Ln>()
    }

    fn exp(&mut self) -> Self {
        self.apply::<Exp>()
    }

    fn sigmoid(&mut self) -> Self {
        self.apply::<Sigmoid>()
    }

    fn tanh(&mut self) -> Self {
        self.apply::<Tanh>()
    }

    fn square(&mut self) -> Self {
        self.apply::<Square>()
    }

    fn abs(&mut self) -> Self {
        self.apply::<Abs>()
    }
}
