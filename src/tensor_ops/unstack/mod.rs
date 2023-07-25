use crate::{shapes::*, tensor::*};

use std::vec::Vec;

mod cpu_kernel;

/// Split a tensor along a dimension into a Vec of tensors
///
/// **Pytorch equivalent** `torch.unbind`.
///
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let tensor: Tensor<Rank3<2, 3, 4>, f32, _> = dev.zeros();
/// let result: Vec<Tensor<Rank2<3, 4>, f32, _>> = tensor.unstack();
/// ```
pub trait TryUnstack: Sized {
    type Unstacked;
    type Err: std::fmt::Debug;

    /// Unstack a tensor along a dimension.
    fn unstack(self) -> Self::Unstacked {
        self.try_unstack().unwrap()
    }
    /// Fallible version of [TryUnstack::unstack]
    fn try_unstack(self) -> Result<Self::Unstacked, Self::Err>;
}

impl<S: Shape, E: Dtype, D: UnstackKernel<E>, T> TryUnstack for Tensor<S, E, D, T>
where
    S: SubDim<Const<N>>,
    T: Tape<E, D>,
{
    type Err = D::Err;
    type Unstacked = Vec<Tensor<S::Smaller, E, D, T>>;

    fn try_unstack(self) -> Result<Self::Unstacked, Self::Err> {
        try_unstack(self)
    }
}

pub trait SubDim<D: Dim>: Shape {
    type Smaller: Shape;
    fn sub_dim(&self, dim: D) -> Self::Smaller;
}

impl<D1: Dim> SubDim<D1> for (D1,) {
    type Smaller = ();

    fn sub_dim(&self, dim: D1) -> Self::Smaller {
        ()
    }
}

pub trait UnstackKernel<E: Dtype>: Storage<E> {
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inp: Tensor<S, E, Self>,
    ) -> Result<Vec<Tensor<S::Smaller, E, Self>>, Self::Err>
    where
        S: SubDim<Num>;
    fn backward(
        &self,
        grad_inp: Vec<&mut Self::Vec>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err>;
}

fn try_unstack<S: Shape, E: Dtype, D: UnstackKernel<E>, T>(
    tensor: Tensor<S, E, D, T>,
) -> Result<Vec<Tensor<S::Smaller, E, D, T>>, D::Err>
where
    S: SubDim<Const<N>>,
    T: Tape<E, D>,
{
    let (input, tape): (Tensor<S, E, D>, T) = tensor.split_tape();
    let device = input.device.clone();
    let tensors = device.forward(input.dim(), input)?;

    let out_ghosts: Vec<_> = tensors.iter().map(|t| t.ghost()).collect();
    let inp_ghost = input.ghost();
    tape.add_backward_op(move |grads| {
        for t in out_ghosts.iter() {
            grads.try_alloc_for(t)?;
        }
        grads.try_alloc_for(&inp_ghost)?;
        let (grad_out, grad_inp) = grads.many_and_ref(&out_ghosts, &inp_ghost);
        device.backward(grad_inp, grad_out)
    });
    Ok(tensors
        .into_iter()
        .map(|t| t.put_tape(tape.clone()))
        .collect())
}
