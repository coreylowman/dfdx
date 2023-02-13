use crate::{shapes::*, tensor::*};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

pub trait Stackable<T>: IntoIterator<Item = T> {
    type Dim: Dim;
    fn dim(&self) -> Self::Dim;
}
impl<T, const N: usize> Stackable<T> for [T; N] {
    type Dim = Const<N>;
    fn dim(&self) -> Self::Dim {
        Const
    }
}
impl<T> Stackable<T> for std::vec::Vec<T> {
    type Dim = usize;
    fn dim(&self) -> Self::Dim {
        self.len()
    }
}

pub trait AddDim<D: Dim>: Shape {
    type Larger: Shape;
}

impl<New: Dim> AddDim<New> for () {
    type Larger = (New,);
}
impl<D1: Dim, New: Dim> AddDim<New> for (D1,) {
    type Larger = (New, D1);
}
impl<D1: Dim, D2: Dim, New: Dim> AddDim<New> for (D1, D2) {
    type Larger = (New, D1, D2);
}
impl<D1: Dim, D2: Dim, D3: Dim, New: Dim> AddDim<New> for (D1, D2, D3) {
    type Larger = (New, D1, D2, D3);
}
impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, New: Dim> AddDim<New> for (D1, D2, D3, D4) {
    type Larger = (New, D1, D2, D3, D4);
}

pub trait TryStackKernel<E: Dtype>: DeviceStorage {
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inp: &[Self::Storage<S, E>],
    ) -> Result<Self::Storage<S::Larger, E>, Self::Err>
    where
        S: AddDim<Num>;
    fn backward<S: Shape, New: Dim>(
        &self,
        num: New,
        inp: &[Self::Storage<S, E>],
        grad_inp: &mut [Self::Storage<S, E>],
        grad_out: &[Self::Storage<S::Larger, E>],
    ) -> Result<(), Self::Err>
    where
        S: AddDim<New>;
}

pub trait TryStack<E: Dtype>: DeviceStorage {
    fn stack<S: Shape, T, Items>(&self, items: Items) -> Tensor<S::Larger, E, Self, T>
    where
        Items: Stackable<Tensor<S, E, Self, T>>,
        S: AddDim<Items::Dim>,
    {
        self.try_stack(items).unwrap()
    }
    fn try_stack<S: Shape, T, Items>(
        &self,
        items: Items,
    ) -> Result<Tensor<S::Larger, E, Self, T>, Self::Err>
    where
        Items: Stackable<Tensor<S, E, Self, T>>,
        S: AddDim<Items::Dim>;
}

impl<E: Dtype, D: DeviceStorage> TryStack<E> for D
where
    Self: TryStackKernel<E>,
{
    fn try_stack<S: Shape, T, Items>(
        &self,
        items: Items,
    ) -> Result<Tensor<S::Larger, E, Self, T>, Self::Err>
    where
        Items: Stackable<Tensor<S, E, Self, T>>,
        S: AddDim<Items::Dim>,
    {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gradients::NoneTape, tensor_ops::*, tests::*};

    #[test]
    fn test_valid_stacks() {
        let dev: TestDevice = Default::default();

        {
            let x: Tensor<(), TestDtype, _> = dev.sample_normal();
            let y: Tensor<(), TestDtype, _> = dev.sample_normal();
            let _: Tensor<Rank1<2>, TestDtype, _> = dev.stack([x, y]);
        }

        {
            let x: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let y: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let z: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
            let _: Tensor<Rank2<3, 3>, TestDtype, _> = dev.stack([x, y, z]);
        }

        {
            let x: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
            let y: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
            let z: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
            let r: Tensor<(usize, Const<2>, Const<3>), TestDtype, _> =
                dev.stack(std::vec![x, y, z]);
            assert_eq!(r.shape().0, 3);
        }
    }

    #[test]
    #[should_panic]
    fn test_stack_with_diff_sizes() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.sample_like(&(2, 3), rand_distr::StandardNormal);
        let y: Tensor<_, TestDtype, _> = dev.sample_like(&(3, 4), rand_distr::StandardNormal);
        let _ = dev.stack([x, y]);
    }

    #[test]
    #[should_panic]
    fn test_stack_with_diff_strides() {
        let dev: TestDevice = Default::default();

        let x: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let y: Tensor<Rank1<3>, TestDtype, _> = dev.sample_normal();
        let _ = dev.stack([x, y.broadcast()]);
    }

    #[test]
    fn test_stack_backwards() {
        let dev: TestDevice = Default::default();

        let x: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let y: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let z: Tensor<Rank2<2, 3>, TestDtype, _> = dev.sample_normal();
        let r = dev.stack([x.trace(), y.trace(), z.trace()]);
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
