use crate::{shapes::*, tensor::*};

use std::vec::Vec;

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

/// Stack an array or vec of tensors together along a new dimension.
pub trait TryStack: Sized {
    type Stacked;
    type Err: std::fmt::Debug;

    /// Stack an array or vec of tensors together along a new dimension.
    ///
    /// An array of tensors will be turned into a [Const] dim, and
    /// a `Vec` of tensors will be turned into a [usize] dim.
    ///
    /// **Pytorch equivalent** `torch.stack`.
    ///
    /// Stacking with an array:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
    /// let b: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
    /// let _: Tensor<Rank3<2, 3, 4>, f32, _> = [a, b].stack();
    /// ```
    ///
    /// Stacking with a vec:
    /// ```rust
    /// # use dfdx::prelude::*;
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
    /// let b: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
    /// let _: Tensor<(usize, Const<3>, Const<4>), f32, _> = vec![a, b].stack();
    /// ```
    fn stack(self) -> Self::Stacked {
        self.try_stack().unwrap()
    }
    /// Fallible version of [TryStack::stack]
    fn try_stack(self) -> Result<Self::Stacked, Self::Err>;
}

impl<S: Shape, E: Dtype, D: StackKernel<E>, T, const N: usize> TryStack for [Tensor<S, E, D, T>; N]
where
    S: AddDim<Const<N>>,
    T: Tape<E, D>,
{
    type Stacked = Tensor<S::Larger, E, D, T>;
    type Err = D::Err;
    fn try_stack(self) -> Result<Self::Stacked, Self::Err> {
        try_stack(self)
    }
}

impl<S: Shape, E: Dtype, D: StackKernel<E>, T> TryStack for std::vec::Vec<Tensor<S, E, D, T>>
where
    S: AddDim<usize>,
    T: Tape<E, D>,
{
    type Stacked = Tensor<S::Larger, E, D, T>;
    type Err = D::Err;
    fn try_stack(self) -> Result<Self::Stacked, Self::Err> {
        try_stack(self)
    }
}

impl<A: TryStack, B: TryStack<Err = A::Err>> TryStack for (A, B) {
    type Stacked = (A::Stacked, B::Stacked);
    type Err = A::Err;
    fn try_stack(self) -> Result<Self::Stacked, Self::Err> {
        Ok((self.0.try_stack()?, self.1.try_stack()?))
    }
}

pub trait AddDim<D: Dim>: Shape {
    type Larger: Shape;
    fn add_dim(&self, dim: D) -> Self::Larger;
}

impl<New: Dim> AddDim<New> for () {
    type Larger = (New,);
    fn add_dim(&self, dim: New) -> Self::Larger {
        (dim,)
    }
}
impl<D1: Dim, New: Dim> AddDim<New> for (D1,) {
    type Larger = (New, D1);
    fn add_dim(&self, dim: New) -> Self::Larger {
        (dim, self.0)
    }
}
impl<D1: Dim, D2: Dim, New: Dim> AddDim<New> for (D1, D2) {
    type Larger = (New, D1, D2);
    fn add_dim(&self, dim: New) -> Self::Larger {
        (dim, self.0, self.1)
    }
}
impl<D1: Dim, D2: Dim, D3: Dim, New: Dim> AddDim<New> for (D1, D2, D3) {
    type Larger = (New, D1, D2, D3);
    fn add_dim(&self, dim: New) -> Self::Larger {
        (dim, self.0, self.1, self.2)
    }
}
impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, New: Dim> AddDim<New> for (D1, D2, D3, D4) {
    type Larger = (New, D1, D2, D3, D4);
    fn add_dim(&self, dim: New) -> Self::Larger {
        (dim, self.0, self.1, self.2, self.3)
    }
}

pub trait StackKernel<E: Dtype>: DeviceStorage {
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inp: &[Tensor<S, E, Self>],
    ) -> Result<Tensor<S::Larger, E, Self>, Self::Err>
    where
        S: AddDim<Num>;
    fn backward(
        &self,
        grad_inp: Vec<&mut Self::Vec<E>>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

fn try_stack<S: Shape, E: Dtype, D: StackKernel<E>, T, Items>(
    items: Items,
) -> Result<Tensor<S::Larger, E, D, T>, D::Err>
where
    Items: Array<Tensor<S, E, D, T>>,
    S: AddDim<Items::Dim>,
    T: Tape<E, D> + Merge<T>,
{
    let new_dim = items.dim();
    assert!(new_dim.size() > 0);

    // need to split tape and transform into Vec for ease of implementation
    let mut tensors = Vec::with_capacity(new_dim.size());
    let mut tape: T = Default::default();
    for item in items.into_iter() {
        let (item, rhs): (Tensor<S, E, D>, T) = item.split_tape();
        tape = tape.merge(rhs);
        tensors.push(item);
    }

    // check that all the shapes are equal
    let device = tensors[0].device.clone();
    let shape = *tensors[0].shape();
    for t in tensors.iter() {
        assert_eq!(t.shape(), &shape);
        tape.try_alloc_grad(t)?;
    }

    // we map to storage refs so kernels don't have to know about tensors
    let out = device.forward(new_dim, &tensors)?;

    let phantom_out = out.clone();
    tape.try_alloc_grad(&out)?;
    tape.add_backward_op(move |grads| {
        let (grad_inp, grad_out) = grads.many_and_ref(&tensors, &phantom_out);
        device.backward(grad_inp, grad_out)?;
        Ok(())
    });
    Ok(out.put_tape(tape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_valid_stacks() {
        let dev: TestDevice = Default::default();

        {
            let x: Tensor<(), TestDtype, _> = dev.sample_normal();
            let y: Tensor<(), TestDtype, _> = dev.sample_normal();
            let _: Tensor<Rank1<2>, TestDtype, _> = [x, y].stack();
        }

        {
            let x: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let y: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let z: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let _: Tensor<Rank2<3, 3>, TestDtype, _> = [x, y, z].stack();
        }

        {
            let x: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
            let y: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
            let z: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
            let r: Tensor<(usize, Const<2>, Const<3>), TestDtype, _> = std::vec![x, y, z].stack();
            assert_eq!(r.shape().0, 3);
        }
    }

    #[test]
    #[should_panic]
    fn test_stack_with_diff_sizes() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.sample_like(&(2, 3), rand_distr::StandardNormal);
        let y: Tensor<_, TestDtype, _> = dev.sample_like(&(3, 4), rand_distr::StandardNormal);
        let _ = [x, y].stack();
    }

    #[test]
    #[should_panic]
    fn test_stack_with_diff_strides() {
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let y: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
        let _ = [x, y.broadcast()].stack();
    }

    #[test]
    fn test_stack_with_all_broadcasted() {
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
        let y: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
        let r = [
            x.trace().broadcast::<Rank2<4, 3>, _>(),
            y.trace().broadcast(),
        ]
        .stack();
        assert_eq!(r.array(), [[x.array(); 4], [y.array(); 4]]);
        let g = r.exp().mean().backward();
        let g1 = [x.trace(), y.trace()].stack().exp().mean().backward();
        assert_eq!(g.get(&x).array(), g1.get(&x).array());
        assert_eq!(g.get(&y).array(), g1.get(&y).array());
    }

    #[test]
    fn test_stack_backwards() {
        let dev: TestDevice = Default::default();

        let x: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let y: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let z: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let r = [x.trace(), y.trace(), z.trace()].stack();
        assert_eq!(r.array(), [x.array(), y.array(), z.array()]);
        let r1 = r.retaped::<NoneTape>();
        let g1 = r1.trace().exp().mean().backward();
        let g = r.exp().mean().backward();
        let r_grad = g1.get(&r1).array();
        assert_eq!(r_grad[0], g.get(&x).array());
        assert_eq!(r_grad[1], g.get(&y).array());
        assert_eq!(r_grad[2], g.get(&z).array());
    }
}
