use crate::{shapes::*, tensor::*};
use std::vec::Vec;

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

#[cfg(feature = "webgpu")]
mod webgpu_kernel;

/// Unstack a tensor along the first dimension into an array or vec of tensors.  
///
/// This is the opposite of [crate::prelude::TryStack].
///
/// A [Const] dim will be turned into an array of tensors, and
/// a [usize] dim will be turned into a `Vec` of tensors.
///
/// **Pytorch equivalent** `torch.unbind` with `dim=0`.
///
/// Unstacking to an array:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let stack: Tensor<Rank3<2, 3, 4>, f32, _> = dev.zeros();
/// let ([a, b], _tape): ([Tensor<Rank2<3, 4>, f32, _>; 2], _) = stack.unstack();
/// ```
///
/// Unstacking to a vec:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let stack: Tensor<(usize, Const::<3>, Const::<4>), f32, _> = dev.zeros_like(&(2, Const, Const));
/// let (unstack, _tape): (Vec<Tensor<Rank2<3, 4>, f32, _>>, _) = stack.unstack();
/// ```
pub trait TryUnstack<Head: Dim>: Sized {
    type Unstacked;

    /// Unstack a tensor along the first dimension into an array or vec of tensors.  
    fn unstack(self) -> Self::Unstacked {
        self.try_unstack().unwrap()
    }
    /// Fallible version of [TryUnstack::unstack]
    fn try_unstack(self) -> Result<Self::Unstacked, Error>;
}

impl<S: Shape, E: Dtype, D: UnstackKernel<E>, T, const N: usize> TryUnstack<Const<N>>
    for Tensor<S, E, D, T>
where
    S: SubDim<Head = Const<N>>,
    D: super::reshape_to::ReshapeKernel<E>,
    T: Tape<E, D>,
{
    type Unstacked = ([Tensor<S::Tail, E, D, T>; N], T);
    fn try_unstack(self) -> Result<Self::Unstacked, Error> {
        try_unstack::<[Option<Tensor<S::Tail, E, D, NoneTape>>; N], _, S, E, D, T>(self)
    }
}

impl<S: Shape, E: Dtype, D: UnstackKernel<E>, T> TryUnstack<usize> for Tensor<S, E, D, T>
where
    S: SubDim<Head = usize>,
    D: super::reshape_to::ReshapeKernel<E>,
    T: Tape<E, D>,
{
    type Unstacked = (Vec<Tensor<S::Tail, E, D, T>>, T);
    fn try_unstack(self) -> Result<Self::Unstacked, Error> {
        try_unstack::<Vec<Option<Tensor<S::Tail, E, D, NoneTape>>>, _, S, E, D, T>(self)
    }
}

pub trait SubDim: Shape {
    type Head: Dim;
    type Tail: Shape;
    fn sub_dim(&self) -> (Self::Head, Self::Tail);
}

impl<D1: Dim> SubDim for (D1,) {
    type Head = D1;
    type Tail = ();
    fn sub_dim(&self) -> (Self::Head, Self::Tail) {
        (self.0, ())
    }
}
impl<D1: Dim, D2: Dim> SubDim for (D1, D2) {
    type Head = D1;
    type Tail = (D2,);
    fn sub_dim(&self) -> (Self::Head, Self::Tail) {
        (self.0, (self.1,))
    }
}
impl<D1: Dim, D2: Dim, D3: Dim> SubDim for (D1, D2, D3) {
    type Head = D1;
    type Tail = (D2, D3);
    fn sub_dim(&self) -> (Self::Head, Self::Tail) {
        (self.0, (self.1, self.2))
    }
}
impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> SubDim for (D1, D2, D3, D4) {
    type Head = D1;
    type Tail = (D2, D3, D4);
    fn sub_dim(&self) -> (Self::Head, Self::Tail) {
        (self.0, (self.1, self.2, self.3))
    }
}
impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> SubDim for (D1, D2, D3, D4, D5) {
    type Head = D1;
    type Tail = (D2, D3, D4, D5);
    fn sub_dim(&self) -> (Self::Head, Self::Tail) {
        (self.0, (self.1, self.2, self.3, self.4))
    }
}
impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim> SubDim for (D1, D2, D3, D4, D5, D6) {
    type Head = D1;
    type Tail = (D2, D3, D4, D5, D6);
    fn sub_dim(&self) -> (Self::Head, Self::Tail) {
        (self.0, (self.1, self.2, self.3, self.4, self.5))
    }
}

pub trait UnstackKernel<E: Dtype>: Storage<E> {
    fn forward<S: Shape, OptionalItems>(
        &self,
        stack: Tensor<S, E, Self, NoneTape>,
    ) -> Result<OptionalItems, Error>
    where
        S: SubDim,
        OptionalItems: Array<Option<Tensor<S::Tail, E, Self, NoneTape>>, Dim = S::Head>;
    fn backward(
        &self,
        grad_stack: &mut Self::Vec,
        grad_unstack: &Self::Vec,
        unstack_idx: usize,
    ) -> Result<(), Error>;
}

fn try_unstack<OptionalItems, Items, S: Shape, E: Dtype, D: UnstackKernel<E>, T>(
    stack: Tensor<S, E, D, T>,
) -> Result<(Items, T), crate::tensor::Error>
where
    S: SubDim,
    T: Tape<E, D>,
    D: super::reshape_to::ReshapeKernel<E>,
    OptionalItems: Array<Option<Tensor<S::Tail, E, D, NoneTape>>, Dim = S::Head>
        + std::ops::IndexMut<usize, Output = Option<Tensor<S::Tail, E, D, NoneTape>>>,
    Items: Array<Tensor<S::Tail, E, D, T>, Dim = S::Head>,
{
    let device = stack.device.clone();
    let (head, _tail) = stack.shape().sub_dim();
    let (stack, stack_tape) = stack.split_tape();

    // TODO: remove this overhead, and panic on a non-contiguous condition
    let stack = {
        use super::reshape_to::ReshapeTo;
        stack.try_contiguous()?
    };

    let stack_ghost = stack.ghost();

    // list of optional tensors (all are Some)

    let mut unstacks = UnstackKernel::forward::<_, OptionalItems>(&device, stack)?;

    // tensors from unstacks must get tapes inserted into them.
    // to do this, from_fn is re-utilized, but this time without optionals
    let unstacks = Items::from_fn(
        |i| {
            let unstack = std::mem::take(&mut unstacks[i]).unwrap();
            let device = device.clone();
            let stack_ghost = stack_ghost.clone();
            let unstack_ghost = unstack.ghost();
            let mut unstack_tape = T::default();

            unstack_tape.add_backward_op(move |grads| {
                grads.try_alloc_for(&stack_ghost)?;
                grads.try_alloc_for(&unstack_ghost)?;
                let (grad_stack, grad_unstack) = grads.mut_and_ref(&stack_ghost, &unstack_ghost);
                UnstackKernel::backward(&device, grad_stack, grad_unstack, i)
            });
            unstack.put_tape(unstack_tape)
        },
        head,
    );

    Ok((unstacks, stack_tape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    // note: based on a stack test
    #[test]
    fn test_valid_unstacks() {
        let dev: TestDevice = Default::default();

        {
            let stack: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let ([_x, _y, _z], _tape): ([Tensor<(), TestDtype, _>; 3], _) = stack.unstack();
        }

        {
            let stack: Tensor<Rank2<3, 2>, TestDtype, _> = dev.sample_normal();
            let ([_x, _y, _z], _tape): ([Tensor<Rank1<2>, TestDtype, _>; 3], _) = stack.unstack();
        }

        {
            let stack: Tensor<(usize, Const<2>), TestDtype, _> =
                dev.sample_normal_like(&(3, Const));
            let (unstacks, _tape): (Vec<Tensor<Rank1<2>, _, _, _>>, _) = stack.unstack();
            assert_eq!(unstacks.len(), 3);
        }
    }

    // note: based on a stack test
    #[test]
    fn test_unstack_backwards() {
        let dev: TestDevice = Default::default();
        let stack: Tensor<Rank2<3, 2>, TestDtype, _> = dev.sample_normal();

        let ([x, y, z], _tape): ([Tensor<Rank1<2>, TestDtype, _, _>; 3], _) =
            stack.leaky_trace().unstack(); // r
        assert_eq!(stack.array(), [x.array(), y.array(), z.array()]);

        let x1 = x.retaped::<NoneTape>(); // r1
        let y1 = y.retaped::<NoneTape>(); // r1
        let z1 = z.retaped::<NoneTape>(); // r1
        let x1g = x1.leaky_trace().exp().mean().backward(); // g1
        let y1g = y1.leaky_trace().exp().mean().backward(); // g1
        let z1g = z1.leaky_trace().exp().mean().backward(); // g1

        let xg = x.exp().mean().backward(); // g
        let yg = y.exp().mean().backward(); // g
        let zg = z.exp().mean().backward(); // g

        let x1_grad = x1g.get(&x1).array(); // r_grad
        let y1_grad = y1g.get(&y1).array(); // r_grad
        let z1_grad = z1g.get(&z1).array(); // r_grad

        assert_eq!(
            [x1_grad, [TestDtype::zero(); 2], [TestDtype::zero(); 2]],
            xg.get(&stack).array()
        );
        assert_eq!(
            [[TestDtype::zero(); 2], y1_grad, [TestDtype::zero(); 2]],
            yg.get(&stack).array()
        );
        assert_eq!(
            [[TestDtype::zero(); 2], [TestDtype::zero(); 2], z1_grad],
            zg.get(&stack).array()
        );

        // extra check
        let stack_g = stack
            .leaky_trace()
            .exp()
            .mean::<_, Axis<1>>()
            .sum()
            .backward();
        assert_eq!(
            stack_g.get(&stack).array(),
            [
                xg.get(&stack).array()[0],
                yg.get(&stack).array()[1],
                zg.get(&stack).array()[2]
            ]
        );
    }
}
