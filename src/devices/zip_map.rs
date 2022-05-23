use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;
use std::ops::*;

/// Zip two Nd arrays together, and apply a generic function to them.
pub trait ZipMapElements<Lhs: CountElements, Rhs: CountElements>: AllocateZeros {
    fn zip_map_into<F: FnMut(&Lhs::Dtype, &Rhs::Dtype) -> Lhs::Dtype + Copy>(
        l: &Lhs,
        r: &Rhs,
        out: &mut Lhs,
        f: F,
    );
    fn zip_map_assign<F: FnMut(&mut Lhs::Dtype, &Rhs::Dtype) + Copy>(l: &mut Lhs, r: &Rhs, f: F);

    fn zip_map<F: FnMut(&Lhs::Dtype, &Rhs::Dtype) -> Lhs::Dtype + Copy>(
        l: &Lhs,
        r: &Rhs,
        f: F,
    ) -> Box<Lhs> {
        let mut out = Self::zeros();
        Self::zip_map_into(l, r, &mut out, f);
        out
    }

    fn add(l: &Lhs, r: &Rhs) -> Box<Lhs>
    where
        for<'r> &'r Lhs::Dtype: Add<&'r Rhs::Dtype, Output = Lhs::Dtype>,
    {
        Self::zip_map(l, r, |x, y| x + y)
    }

    fn sub(l: &Lhs, r: &Rhs) -> Box<Lhs>
    where
        for<'r> &'r Lhs::Dtype: Sub<&'r Rhs::Dtype, Output = Lhs::Dtype>,
    {
        Self::zip_map(l, r, |x, y| x - y)
    }

    fn mul(l: &Lhs, r: &Rhs) -> Box<Lhs>
    where
        for<'r> &'r Lhs::Dtype: Mul<&'r Rhs::Dtype, Output = Lhs::Dtype>,
    {
        Self::zip_map(l, r, |x, y| x * y)
    }

    fn div(l: &Lhs, r: &Rhs) -> Box<Lhs>
    where
        for<'r> &'r Lhs::Dtype: Div<&'r Rhs::Dtype, Output = Lhs::Dtype>,
    {
        Self::zip_map(l, r, |x, y| x / y)
    }

    fn add_assign(l: &mut Lhs, r: &Rhs)
    where
        for<'r> Lhs::Dtype: AddAssign<&'r Rhs::Dtype>,
    {
        Self::zip_map_assign(l, r, |x, y| *x += y);
    }

    fn sub_assign(l: &mut Lhs, r: &Rhs)
    where
        for<'r> Lhs::Dtype: SubAssign<&'r Rhs::Dtype>,
    {
        Self::zip_map_assign(l, r, |x, y| *x -= y);
    }

    fn mul_assign(l: &mut Lhs, r: &Rhs)
    where
        for<'r> Lhs::Dtype: MulAssign<&'r Rhs::Dtype>,
    {
        Self::zip_map_assign(l, r, |x, y| *x *= y);
    }

    fn div_assign(l: &mut Lhs, r: &Rhs)
    where
        for<'r> Lhs::Dtype: DivAssign<&'r Rhs::Dtype>,
    {
        Self::zip_map_assign(l, r, |x, y| *x /= y);
    }
}

impl ZipMapElements<f32, f32> for Cpu {
    fn zip_map_into<F: FnMut(&f32, &f32) -> f32 + Copy>(l: &f32, r: &f32, out: &mut f32, mut f: F) {
        *out = f(l, r);
    }

    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(l: &mut f32, r: &f32, mut f: F) {
        f(l, r)
    }
}

impl<const M: usize> ZipMapElements<[f32; M], f32> for Cpu {
    fn zip_map_into<F: FnMut(&f32, &f32) -> f32 + Copy>(
        l: &[f32; M],
        r: &f32,
        out: &mut [f32; M],
        f: F,
    ) {
        for i in 0..M {
            Self::zip_map_into(&l[i], r, &mut out[i], f);
        }
    }

    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(l: &mut [f32; M], r: &f32, f: F) {
        for i in 0..M {
            Self::zip_map_assign(&mut l[i], r, f);
        }
    }
}

impl<Rhs, Lhs, const M: usize> ZipMapElements<[Lhs; M], [Rhs; M]> for Cpu
where
    Lhs: CountElements,
    Rhs: CountElements,
    Self: ZipMapElements<Lhs, Rhs>,
{
    fn zip_map_into<F: FnMut(&Lhs::Dtype, &Rhs::Dtype) -> Lhs::Dtype + Copy>(
        l: &[Lhs; M],
        r: &[Rhs; M],
        out: &mut [Lhs; M],
        f: F,
    ) {
        for i in 0..M {
            Self::zip_map_into(&l[i], &r[i], &mut out[i], f);
        }
    }

    fn zip_map_assign<F: FnMut(&mut Lhs::Dtype, &Rhs::Dtype) + Copy>(
        l: &mut [Lhs; M],
        r: &[Rhs; M],
        f: F,
    ) {
        for i in 0..M {
            Self::zip_map_assign(&mut l[i], &r[i], f);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::ZeroElements;

    use super::*;

    #[test]
    fn test_0d_zip_map() {
        let a = 1.0;
        let b = 2.0;
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x + y);
        assert_eq!(c, 3.0);
    }

    #[test]
    fn test_1d_zip_map() {
        let a = [1.0; 3];
        let b = [2.0; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x + y);
        assert_eq!(c, [3.0; 3]);
    }

    #[test]
    fn test_1d_zip_map_broadcast_inner() {
        let a = [1.0; 3];
        let b = 2.0;
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x + y);
        assert_eq!(c, [3.0; 3]);
    }

    #[test]
    fn test_2d_zip_map() {
        let a = [[2.0; 2]; 3];
        let b = [[3.0; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x * y);
        assert_eq!(c, [[6.0; 2]; 3]);
    }

    #[test]
    fn test_2d_zip_map_broadcast_inner() {
        let a = [[2.0; 2]; 3];
        let b = [3.0; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x * y);
        assert_eq!(c, [[6.0; 2]; 3]);
    }

    #[test]
    fn test_3d_zip_map() {
        let a = [[[2.0; 5]; 2]; 3];
        let b = [[[3.0; 5]; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x * y);
        assert_eq!(c, [[[6.0; 5]; 2]; 3]);
    }

    #[test]
    fn test_3d_zip_map_broadcast_inner() {
        let a = [[[2.0; 5]; 2]; 3];
        let b = [[3.0; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x * y);
        assert_eq!(c, [[[6.0; 5]; 2]; 3]);
    }
}
