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
    S: SubDim,
    T: Tape<E, D>,
{
    type Err = D::Err;
    type Unstacked = Vec<Tensor<S::Smaller, E, D, T>>;

    fn try_unstack(self) -> Result<Self::Unstacked, Self::Err> {
        try_unstack(self)
    }
}

pub trait SubDim: Shape {
    type Smaller: Shape;
    fn sub_dim(&self) -> Self::Smaller;
}

// impl<const N: usize> SubDim<Const<N>> for (Const<N>,) {
//     type Smaller = ();

//     fn sub_dim(&self) -> Self::Smaller {
//         ()
//     }
// }

impl<D1: Dim> SubDim for (D1,) {
    type Smaller = ();
    fn sub_dim(&self) -> Self::Smaller {
        ()
    }
}

impl<D1: Dim, D2: Dim> SubDim for (D1, D2) {
    type Smaller = (D2,);
    fn sub_dim(&self) -> Self::Smaller {
        (self.1,)
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> SubDim for (D1, D2, D3) {
    type Smaller = (D2, D3);
    fn sub_dim(&self) -> Self::Smaller {
        (self.1, self.2)
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> SubDim for (D1, D2, D3, D4) {
    type Smaller = (D2, D3, D4);
    fn sub_dim(&self) -> Self::Smaller {
        (self.1, self.2, self.3)
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> SubDim for (D1, D2, D3, D4, D5) {
    type Smaller = (D2, D3, D4, D5);
    fn sub_dim(&self) -> Self::Smaller {
        (self.1, self.2, self.3, self.4)
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim> SubDim for (D1, D2, D3, D4, D5, D6) {
    type Smaller = (D2, D3, D4, D5, D6);
    fn sub_dim(&self) -> Self::Smaller {
        (self.1, self.2, self.3, self.4, self.5)
    }
}

// impl<const N: usize> SubDim for (Const<N>,) {
//     type Smaller = ();

//     fn sub_dim(&self) -> Self::Smaller {
//         ()
//     }
// }

pub trait UnstackKernel<E: Dtype>: Storage<E> {
    fn forward<S: Shape>(
        &self,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Vec<Tensor<S::Smaller, E, Self>>, Self::Err>
    where
        S: SubDim;
    fn backward(
        &self,
        grad_inp: &mut Self::Vec,
        grad_out: Vec<&Self::Vec>,
    ) -> Result<(), Self::Err>;
}

fn try_unstack<S: Shape, E: Dtype, D: UnstackKernel<E>, T: Tape<E, D>>(
    tensor: Tensor<S, E, D, T>,
) -> Result<Vec<Tensor<S::Smaller, E, D, T>>, D::Err>
where
    S: SubDim,
{
    let (input, mut tape): (Tensor<S, E, D>, T) = tensor.split_tape();
    let device = input.device.clone();
    let tensors = device.forward(&input)?;

    let out_ghosts: Vec<_> = tensors.iter().map(|t| t.ghost()).collect();
    let inp_ghost = input.ghost();
    tape.add_backward_op(move |grads| {
        for t in out_ghosts.iter() {
            grads.try_alloc_for(t)?;
        }
        grads.try_alloc_for(&inp_ghost)?;
        let (grad_out, grad_inp) = grads.many_mut_and_ref(&out_ghosts, &inp_ghost);
        device.backward(grad_inp, grad_out)
    });

    let mut tensors = tensors.into_iter();
    let first = tensors.next().map(|t| t.put_tape(tape));
    let others = tensors
        .map(|t| t.put_tape(Default::default()))
        .collect::<Vec<_>>();

    let mut result = Vec::new();
    if let Some(first) = first {
        result.push(first);
    }
    result.extend(others);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_valid_unstacks() {
        let dev: TestDevice = Default::default();

        {
            let stacked: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let unstacked = stacked.clone().unstack();
            assert_eq!(unstacked.len(), 3);
            for (index, item) in unstacked.into_iter().enumerate() {
                assert_eq!(item.shape(), &());
                assert_eq!(item.data[0], stacked.data[index]);
            }
        }

        {
            let stacked: Tensor<Rank2<4, 3>, TestDtype, _> = dev.sample_normal();
            let unstacked = stacked.clone().unstack();
            assert_eq!(unstacked.len(), 4);
            for (index, item) in unstacked.into_iter().enumerate() {
                assert_eq!(item.shape(), &(Const::<3>,));
                // assert_eq!(item.data[0], stacked.data[index * 3]);
                for (i, &value) in item.data.iter().enumerate() {
                    assert_eq!(value, stacked.data[index * 3 + i]);
                }
            }
        }
    }
}
