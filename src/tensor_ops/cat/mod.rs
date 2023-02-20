use crate::{shapes::*, tensor::*};

/// Concatenate two tensors along the first dimension.
pub trait TryConcat<Rhs: Shape>: DeviceStorage {
    /// Concatenate two tensors along the first dimension.
    ///
    /// TODO
    fn concat(&self, rhs: &Rhs);

    /// Fallible version of [TryConcat::concat].
    fn try_concat(&self, rhs: &Rhs) -> Result<(), Self::Err>;
}

pub trait ExtendDim<Rhs: Shape>: Shape {
    type Extended: Shape;
    fn extend(&self, rhs: &Rhs) -> Self::Extended;
}

macro_rules! extend {
    ($d1:ident, $($dn:ident),*) => {
        impl<$d1: Dim, $($dn: Dim),*> ExtendDim<($d1, $($dn),*)> for (usize, $($dn),*) {
            type Extended = (usize, $($dn),*);

            fn extend(&self, rhs: &($d1, $($dn),*)) -> Self::Extended {
                let mut shape = *self.shape();
                shape.0 += rhs.0.size();
                shape
            }
        }
    };
}
extend!(D1,);
extend!(D1, D2);
extend!(D1, D2, D3);
extend!(D1, D2, D3, D4);
extend!(D1, D2, D3, D4, D5);
extend!(D1, D2, D3, D4, D5, D6);
