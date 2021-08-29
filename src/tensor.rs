use crate::{
    gradients::{GradientRef, GradientTape, Op},
    traits::{Params, Tensor},
};
use ndarray::prelude::*;
use ndarray_rand::rand::Rng;
use std::ops::{Add, Mul, Sub};

#[derive(Default, Debug)]
pub struct Tensor0D {
    data: Array0<f32>,
    grad: GradientRef,
}

impl Params for Tensor0D {
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.data.map_inplace(|f| *f = rng.gen())
    }

    fn register(&mut self, tape: &mut GradientTape) {
        if !self.grad.has_tag() {
            self.set_tag(Some(tape.advance(self.shape())));
        }
    }

    fn update(&mut self, tape: &GradientTape) {
        let gradient = &tape[self.grad().tag()];
        self.data -= gradient;
        self.set_tag(None);
    }
}

impl Tensor for Tensor0D {
    type Dimension = Ix0;

    fn shape(&self) -> &[usize] {
        &[]
    }

    fn grad(&self) -> &GradientRef {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut GradientRef {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

#[derive(Debug)]
pub struct Tensor1D<const N: usize> {
    data: Array1<f32>,
    grad: GradientRef,
}

impl<const N: usize> Default for Tensor1D<N> {
    fn default() -> Self {
        Self {
            data: Array1::<f32>::zeros((N,)),
            grad: Default::default(),
        }
    }
}

impl<const N: usize> Params for Tensor1D<N> {
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.data.map_inplace(|f| *f = rng.gen());
    }

    fn register(&mut self, tape: &mut GradientTape) {
        if !self.grad.has_tag() {
            self.set_tag(Some(tape.advance(self.shape())));
        }
    }

    fn update(&mut self, tape: &GradientTape) {
        let gradient = &tape[self.grad().tag()];
        self.data -= gradient;
        self.set_tag(None);
    }
}

impl<const N: usize> Tensor for Tensor1D<N> {
    type Dimension = Ix1;

    fn shape(&self) -> &[usize] {
        &[N]
    }

    fn grad(&self) -> &GradientRef {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut GradientRef {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const N: usize> Add for &mut Tensor1D<N> {
    type Output = Tensor1D<N>;
    fn add(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let mut result = Tensor1D {
            data: &self.data + &rhs.data,
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape().or(rhs.take_tape());
        result.set_tag(opt_tape.as_mut().map(|tape| {
            self.register(tape);
            rhs.register(tape);

            tape.binary_op(
                Op::Add,
                self.grad.tag(),
                rhs.grad.tag(),
                Array1::<f32>::ones((N,)).into_dyn(),
                Array1::<f32>::ones((N,)).into_dyn(),
                &[N],
            )
        }));
        result.keep_tape(opt_tape);

        result
    }
}

impl<const N: usize> Sub for &mut Tensor1D<N> {
    type Output = Tensor1D<N>;
    fn sub(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let mut result = Tensor1D {
            data: &self.data - &rhs.data,
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape().or(rhs.take_tape());
        result.set_tag(opt_tape.as_mut().map(|tape| {
            self.register(tape);
            rhs.register(tape);

            tape.binary_op(
                Op::Sub,
                self.grad.tag(),
                rhs.grad.tag(),
                Array1::<f32>::ones((N,)).into_dyn(),
                Array1::<f32>::ones((N,)).into_dyn(),
                &[N],
            )
        }));
        result.keep_tape(opt_tape);

        result
    }
}

impl<const N: usize> Tensor1D<N> {
    pub fn square(&mut self) -> Tensor1D<N> {
        let mut result = Tensor1D {
            data: self.data.map(|f| f.powi(2)),
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape();
        result.set_tag(opt_tape.as_mut().map(|tape| {
            let deriv = 2.0 * &self.data;
            tape.unary_op(Op::Square, self.grad.tag(), deriv.into_dyn(), &[N])
        }));
        result.keep_tape(opt_tape);

        result
    }

    pub fn mean(&mut self) -> Tensor0D {
        let mut result = Tensor0D {
            data: arr0(self.data.mean().unwrap()),
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape();
        result.set_tag(opt_tape.as_mut().map(|tape| {
            let deriv = 2.0 * &Array1::<f32>::ones((N,));
            tape.unary_op(Op::Mean, self.grad.tag(), deriv.into_dyn(), &[])
        }));
        result.keep_tape(opt_tape);

        result
    }
}

#[derive(Debug)]
pub struct Tensor2D<const M: usize, const N: usize> {
    data: Array2<f32>,
    grad: GradientRef,
}

impl<const M: usize, const N: usize> Default for Tensor2D<M, N> {
    fn default() -> Self {
        Self {
            data: Array2::<f32>::zeros((M, N)),
            grad: Default::default(),
        }
    }
}

impl<const M: usize, const N: usize> Params for Tensor2D<M, N> {
    fn randomize<R: Rng>(&mut self, rng: &mut R) {
        self.data.map_inplace(|f| *f = rng.gen());
    }

    fn register(&mut self, tape: &mut GradientTape) {
        if !self.grad.has_tag() {
            self.set_tag(Some(tape.advance(self.shape())));
        }
    }

    fn update(&mut self, tape: &GradientTape) {
        let gradient = &tape[self.grad().tag()];
        self.data -= gradient;
        self.set_tag(None);
    }
}

impl<const M: usize, const N: usize> Tensor for Tensor2D<M, N> {
    type Dimension = Ix2;

    fn shape(&self) -> &[usize] {
        &[M, N]
    }

    fn grad(&self) -> &GradientRef {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut GradientRef {
        &mut self.grad
    }

    fn data(&self) -> &Array<f32, Self::Dimension> {
        &self.data
    }

    fn mut_data(&mut self) -> &mut Array<f32, Self::Dimension> {
        &mut self.data
    }
}

impl<const M: usize, const N: usize> Mul<&mut Tensor1D<N>> for &mut Tensor2D<M, N> {
    type Output = Tensor1D<M>;
    fn mul(self, rhs: &mut Tensor1D<N>) -> Self::Output {
        let mut result = Tensor1D {
            data: self.data.dot(&rhs.data),
            grad: Default::default(),
        };

        let mut opt_tape = self.take_tape().or(rhs.take_tape());
        result.set_tag(opt_tape.as_mut().map(|tape| {
            self.register(tape);
            rhs.register(tape);

            let lhs_deriv = rhs.data.clone().into_shape((N, 1)).expect("");
            let rhs_deriv = self.data.clone().reversed_axes();

            tape.binary_op(
                Op::Matmul { m: M, n: N },
                self.grad.tag(),
                rhs.grad.tag(),
                lhs_deriv.into_dyn(),
                rhs_deriv.into_dyn(),
                &[M],
            )
        }));
        result.keep_tape(opt_tape);

        result
    }
}
