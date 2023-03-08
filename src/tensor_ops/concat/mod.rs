use crate::{
    gradients::{Merge, Tape},
    shapes::*,
    tensor::*,
};

// / Concatenate two tensors along the first dimension.
// pub trait TryConcat<E: Dtype>: DeviceStorage {
//     /// Concatenate two tensors along the first dimension.
//     ///
//     /// TODO
//     fn concat<A: Shape, B: Shape, T>(
//         &self,
//         a: Tensor<A, E, Self, T>,
//         b: Tensor<B, E, Self, T>,
//     ) -> Tensor<A::Extended, E, Self, T>
//     where
//         A: ConcatShape<B>,
//         T: Tape<Self> + Merge<T>,
//     {
//         self.try_concat(a, b).unwrap()
//     }

//     /// Fallible version of [TryConcat::concat].
//     fn try_concat<A: Shape, B: Shape, T>(
//         &self,
//         a: Tensor<A, E, Self, T>,
//         b: Tensor<B, E, Self, T>,
//     ) -> Result<Tensor<A::Extended, E, Self, T>, Self::Err>
//     where
//         A: ConcatShape<B>,
//         T: Tape<Self> + Merge<T>;
// }

pub trait ConcatShape<Rhs: Shape>: Shape {
    type Catted: Shape;
    fn concat_shape(&self, rhs: &Rhs) -> Self::Catted;
}

impl ConcatShape<()> for () {
    type Catted = ();
    fn concat_shape(&self, _: &()) -> Self::Catted {}
}

macro_rules! impl_concat {
    ([$($Dims:tt $Idx:tt),*]) => {
impl<A: Dim, B: Dim, $($Dims: Dim, )*> ConcatShape<(A, $($Dims, )*)>
    for (B, $($Dims, )*)
where
    A: std::ops::Add<B>,
    <A as std::ops::Add<B>>::Output: Dim,
{
    type Catted = (<A as std::ops::Add<B>>::Output, $($Dims, )*);

    fn concat_shape(&self, rhs: &(A, $($Dims, )*)) -> Self::Catted {
        $(assert_eq!(self.$Idx, rhs.$Idx);)*
        (rhs.0 + self.0, $(self.$Idx, )*)
    }
}
    };
}

impl_concat!([]);
impl_concat!([D1 1]);
impl_concat!([D1 1, D2 2]);
impl_concat!([D1 1, D2 2, D3 3]);
impl_concat!([D1 1, D2 2, D3 3, D4 4]);
impl_concat!([D1 1, D2 2, D3 3, D4 4, D5 5]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_concat_shape() {
        let a: (usize, Const<5>) = (5, Const);
        let b: (usize, Const<5>) = (3, Const);
        assert_eq!(a.concat_shape(&b), (8, Const::<5>));

        let a: (Const<5>, Const<5>) = (Const, Const);
        let b: (usize, Const<5>) = (3, Const);
        assert_eq!(a.concat_shape(&b), (8, Const::<5>));

        let a: (usize, Const<5>) = (5, Const);
        let b: (Const<3>, Const<5>) = (Const, Const);
        assert_eq!(a.concat_shape(&b), (8, Const::<5>));

        #[cfg(feature = "nightly")]
        {
            let a: (Const<5>, Const<5>) = (Const, Const);
            let b: (Const<3>, Const<5>) = (Const, Const);
            assert_eq!(a.concat_shape(&b), (Const::<8>, Const::<5>));
        }
    }
}
