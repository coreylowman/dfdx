use crate::{
    gradients::{Merge, Tape},
    shapes::*,
    tensor::*,
};

/// Concatenate two tensors along the first dimension.
pub trait TryConcat<E: Dtype>: DeviceStorage {
    /// Concatenate two tensors along the first dimension.
    ///
    /// TODO
    fn concat<A: Shape, B: Shape, T>(
        &self,
        a: Tensor<A, E, Self, T>,
        b: Tensor<B, E, Self, T>,
    ) -> Tensor<A::Extended, E, Self, T>
    where
        A: ExtendDim<B>,
        T: Tape<Self> + Merge<T>,
    {
        self.try_concat(a, b).unwrap()
    }

    /// Fallible version of [TryConcat::concat].
    fn try_concat<A: Shape, B: Shape, T>(
        &self,
        a: Tensor<A, E, Self, T>,
        b: Tensor<B, E, Self, T>,
    ) -> Result<Tensor<A::Extended, E, Self, T>, Self::Err>
    where
        A: ExtendDim<B>,
        T: Tape<Self> + Merge<T>;
}

pub trait ExtendDim<Rhs: Shape>: Shape {
    type Extended: Shape;
    fn extend_dim(&self, rhs: &Rhs) -> Self::Extended;
}

#[rustfmt::skip]
impl<A: Dim, B: Dim>
    ExtendDim<(A,)> for (B,)
where
    A: std::ops::Add<B>,
    <A as std::ops::Add<B>>::Output: Dim,
{
    type Extended = (<A as std::ops::Add<B>>::Output,);

    fn extend_dim(&self, rhs: &(A,)) -> Self::Extended {
        (rhs.0 + self.0,)
    }
}
#[rustfmt::skip]
impl<A: Dim, B: Dim, D2: Dim>
    ExtendDim<(A, D2)> for (B, D2)
where
    A: std::ops::Add<B>,
    <A as std::ops::Add<B>>::Output: Dim,
{
    type Extended = (<A as std::ops::Add<B>>::Output, D2);

    fn extend_dim(&self, rhs: &(A, D2)) -> Self::Extended {
        (rhs.0 + self.0, self.1)
    }
}
#[rustfmt::skip]
impl<A: Dim, B: Dim, D2: Dim, D3: Dim>
    ExtendDim<(A, D2, D3)> for (B, D2, D3)
where
    A: std::ops::Add<B>,
    <A as std::ops::Add<B>>::Output: Dim,
{
    type Extended = (<A as std::ops::Add<B>>::Output, D2, D3);

    fn extend_dim(&self, rhs: &(A, D2, D3)) -> Self::Extended {
        (rhs.0 + self.0, self.1, self.2)
    }
}
#[rustfmt::skip]
impl<A: Dim, B: Dim, D2: Dim, D3: Dim, D4: Dim>
    ExtendDim<(A, D2, D3, D4)> for (B, D2, D3, D4)
where
    A: std::ops::Add<B>,
    <A as std::ops::Add<B>>::Output: Dim,
{
    type Extended = (<A as std::ops::Add<B>>::Output, D2, D3, D4);

    fn extend_dim(&self, rhs: &(A, D2, D3, D4)) -> Self::Extended {
        (rhs.0 + self.0, self.1, self.2, self.3)
    }
}
#[rustfmt::skip]
impl<A: Dim, B: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim>
    ExtendDim<(A, D2, D3, D4, D5)> for (B, D2, D3, D4, D5)
where
    A: std::ops::Add<B>,
    <A as std::ops::Add<B>>::Output: Dim,
{
    type Extended = (<A as std::ops::Add<B>>::Output, D2, D3, D4, D5);

    fn extend_dim(&self, rhs: &(A, D2, D3, D4, D5)) -> Self::Extended {
        (rhs.0 + self.0, self.1, self.2, self.3, self.4)
    }
}
#[rustfmt::skip]
impl<A: Dim, B: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim>
    ExtendDim<(A, D2, D3, D4, D5, D6)> for (B, D2, D3, D4, D5, D6)
where
    A: std::ops::Add<B>,
    <A as std::ops::Add<B>>::Output: Dim,
{
    type Extended = (<A as std::ops::Add<B>>::Output, D2, D3, D4, D5, D6);

    fn extend_dim(&self, rhs: &(A, D2, D3, D4, D5, D6)) -> Self::Extended {
        (rhs.0 + self.0, self.1, self.2, self.3, self.4, self.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_concat() {
        let dev: TestDevice = Default::default();

        let a: Tensor<(usize, Const<2>), TestDtype, _> = dev.zeros_like(&(3, Const));
        let b: Tensor<(usize, Const<2>), TestDtype, _> = dev.zeros_like(&(5, Const));

        a.shape().extend_dim(b.shape());
        todo!()
    }
}
