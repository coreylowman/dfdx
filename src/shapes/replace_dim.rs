use super::shape::{Dim, Shape};

pub trait ReplaceDim<const I: isize, New: Dim>: Shape {
    type Replaced: Shape<Concrete = Self::Concrete>;
    fn replace(&self, dim: New) -> Self::Replaced {
        let mut c = self.concrete();
        c[I as usize] = dim.size();
        Self::Replaced::from_concrete(&c).unwrap()
    }
}

macro_rules! impl_replace {
    (($($DimVars:tt),*), $Ax:tt, $New:ty) => {
impl<$($DimVars: Dim, )* New: Dim> ReplaceDim<$Ax, New> for ($($DimVars, )*) {
    type Replaced = $New;
}
    };
}

impl_replace!((D1), 0, (New,));

impl_replace!((D1, D2), 0, (New, D2));
impl_replace!((D1, D2), 1, (D1, New));

impl_replace!((D1, D2, D3), 0, (New, D2, D3));
impl_replace!((D1, D2, D3), 1, (D1, New, D3));
impl_replace!((D1, D2, D3), 2, (D1, D2, New));

impl_replace!((D1, D2, D3, D4), 0, (New, D2, D3, D4));
impl_replace!((D1, D2, D3, D4), 1, (D1, New, D3, D4));
impl_replace!((D1, D2, D3, D4), 2, (D1, D2, New, D4));
impl_replace!((D1, D2, D3, D4), 3, (D1, D2, D3, New));

impl_replace!((D1, D2, D3, D4, D5), 0, (New, D2, D3, D4, D5));
impl_replace!((D1, D2, D3, D4, D5), 1, (D1, New, D3, D4, D5));
impl_replace!((D1, D2, D3, D4, D5), 2, (D1, D2, New, D4, D5));
impl_replace!((D1, D2, D3, D4, D5), 3, (D1, D2, D3, New, D5));
impl_replace!((D1, D2, D3, D4, D5), 4, (D1, D2, D3, D4, New));

impl_replace!((D1, D2, D3, D4, D5, D6), 0, (New, D2, D3, D4, D5, D6));
impl_replace!((D1, D2, D3, D4, D5, D6), 1, (D1, New, D3, D4, D5, D6));
impl_replace!((D1, D2, D3, D4, D5, D6), 2, (D1, D2, New, D4, D5, D6));
impl_replace!((D1, D2, D3, D4, D5, D6), 3, (D1, D2, D3, New, D5, D6));
impl_replace!((D1, D2, D3, D4, D5, D6), 4, (D1, D2, D3, D4, New, D6));
impl_replace!((D1, D2, D3, D4, D5, D6), 5, (D1, D2, D3, D4, D5, New));
