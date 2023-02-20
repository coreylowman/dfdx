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
    fn extend(&self, rhs: &Rhs) -> Self::Extended;
}

#[rustfmt::skip]
impl ExtendDim<(usize,)> for (usize,) {
    type Extended = (usize,);
    fn extend(&self, rhs: &(usize,)) -> Self::Extended {
        let s = self.shape();
        (s.0.size() + rhs.0.size(),)
    }
}
#[rustfmt::skip]
impl<D2: Dim>
    ExtendDim<(usize, D2)> for (usize, D2)
{
    type Extended = (usize, D2);
    fn extend(&self, rhs: &(usize, D2)) -> Self::Extended {
        let s = self.shape();
        (s.0.size() + rhs.0.size(), s.1)
    }
}
#[rustfmt::skip]
impl<D2: Dim, D3: Dim>
    ExtendDim<(usize, D2, D3)> for (usize, D2, D3)
{
    type Extended = (usize, D2, D3);
    fn extend(&self, rhs: &(usize, D2, D3)) -> Self::Extended {
        let s = self.shape();
        (s.0.size() + rhs.0.size(), s.1, s.2)
    }
}
#[rustfmt::skip]
impl<D2: Dim, D3: Dim, D4: Dim>
    ExtendDim<(usize, D2, D3, D4)> for (usize, D2, D3, D4)
{
    type Extended = (usize, D2, D3, D4);
    fn extend(&self, rhs: &(usize, D2, D3, D4)) -> Self::Extended {
        let s = self.shape();
        (s.0.size() + rhs.0.size(), s.1, s.2, s.3)
    }
}
#[rustfmt::skip]
impl<D2: Dim, D3: Dim, D4: Dim, D5: Dim>
    ExtendDim<(usize, D2, D3, D4, D5)> for (usize, D2, D3, D4, D5)
{
    type Extended = (usize, D2, D3, D4, D5);
    fn extend(&self, rhs: &(usize, D2, D3, D4, D5)) -> Self::Extended {
        let s = self.shape();
        (s.0.size() + rhs.0.size(), s.1, s.2, s.3, s.4)
    }
}
#[rustfmt::skip]
impl<D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim>
    ExtendDim<(usize, D2, D3, D4, D5, D6)> for (usize, D2, D3, D4, D5, D6)
{
    type Extended = (usize, D2, D3, D4, D5, D6);
    fn extend(&self, rhs: &(usize, D2, D3, D4, D5, D6)) -> Self::Extended {
        let s = self.shape();
        (s.0.size() + rhs.0.size(), s.1, s.2, s.3, s.4, s.5)
    }
}

#[rustfmt::skip]
#[cfg(feature = "nightly")]
impl<const D: usize, const D1: usize>
    ExtendDim<(Const<D1>,)> for (Const<D>,)
where
    Const<{ D + D1 }>: Sized,
{
    type Extended = (Const<{ D + D1 }>,);
    fn extend(&self, _rhs: &(Const<D1>,)) -> Self::Extended {
        (Const,)
    }
}
#[rustfmt::skip]
#[cfg(feature = "nightly")]
impl<const D: usize, const D1: usize, D2: Dim>
    ExtendDim<(Const<D1>, D2)> for (Const<D>, D2)
where
    Const<{ D + D1 }>: Sized,
{
    type Extended = (Const<{ D + D1 }>, D2);
    fn extend(&self, _rhs: &(Const<D1>, D2)) -> Self::Extended {
        let s = self.shape();
        (Const, s.1)
    }
}
#[rustfmt::skip]
#[cfg(feature = "nightly")]
impl<const D: usize, const D1: usize, D2: Dim, D3: Dim>
    ExtendDim<(Const<D1>, D2, D3)> for (Const<D>, D2, D3)
where
    Const<{ D + D1 }>: Sized,
{
    type Extended = (Const<{ D + D1 }>, D2, D3);
    fn extend(&self, _rhs: &(Const<D1>, D2, D3)) -> Self::Extended {
        let s = self.shape();
        (Const, s.1, s.2)
    }
}
#[rustfmt::skip]
#[cfg(feature = "nightly")]
impl<const D: usize, const D1: usize, D2: Dim, D3: Dim, D4: Dim>
    ExtendDim<(Const<D1>, D2, D3, D4)> for (Const<D>, D2, D3, D4)
where
    Const<{ D + D1 }>: Sized,
{
    type Extended = (Const<{ D + D1 }>, D2, D3, D4);
    fn extend(&self, _rhs: &(Const<D1>, D2, D3, D4)) -> Self::Extended {
        let s = self.shape();
        (Const, s.1, s.2, s.3)
    }
}
#[rustfmt::skip]
#[cfg(feature = "nightly")]
impl<const D: usize, const D1: usize, D2: Dim, D3: Dim, D4: Dim, D5: Dim>
    ExtendDim<(Const<D1>, D2, D3, D4, D5)> for (Const<D>, D2, D3, D4, D5)
where
    Const<{ D + D1 }>: Sized,
{
    type Extended = (Const<{ D + D1 }>, D2, D3, D4, D5);
    fn extend(&self, _rhs: &(Const<D1>, D2, D3, D4, D5)) -> Self::Extended {
        let s = self.shape();
        (Const, s.1, s.2, s.3, s.4)
    }
}
#[rustfmt::skip]
#[cfg(feature = "nightly")]
impl<const D: usize, const D1: usize, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim>
    ExtendDim<(Const<D1>, D2, D3, D4, D5, D6)> for (Const<D>, D2, D3, D4, D5, D6)
where
    Const<{ D + D1 }>: Sized,
{
    type Extended = (Const<{ D + D1 }>, D2, D3, D4, D5, D6);
    fn extend(&self, _rhs: &(Const<D1>, D2, D3, D4, D5, D6)) -> Self::Extended {
        let s = self.shape();
        (Const, s.1, s.2, s.3, s.4, s.5)
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

        a.shape().extend(b.shape());
        todo!()
    }
}
