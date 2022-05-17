use super::{AllocateZeros, CountElements, Cpu};

pub trait ZipMapElements<Lhs, Rhs> {
    type Output: CountElements;

    fn zip_map_into<F: FnMut(&f32, &f32) -> f32 + Copy>(
        l: &Lhs,
        r: &Rhs,
        out: &mut Self::Output,
        f: F,
    );
    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(l: &mut Lhs, r: &Rhs, f: F);

    fn zip_map<F: FnMut(&f32, &f32) -> f32 + Copy>(l: &Lhs, r: &Rhs, f: F) -> Box<Self::Output>
    where
        Self: AllocateZeros<Self::Output>,
    {
        let mut out = Self::zeros();
        Self::zip_map_into(l, r, &mut out, f);
        out
    }

    fn add(l: &Lhs, r: &Rhs) -> Box<Self::Output>
    where
        Self: AllocateZeros<Self::Output>,
    {
        Self::zip_map(l, r, |x, y| x + y)
    }

    fn add_assign(l: &mut Lhs, r: &Rhs) {
        Self::zip_map_assign(l, r, |x, y| *x += y);
    }

    fn sub(l: &Lhs, r: &Rhs) -> Box<Self::Output>
    where
        Self: AllocateZeros<Self::Output>,
    {
        Self::zip_map(l, r, |x, y| x - y)
    }

    fn sub_assign(l: &mut Lhs, r: &Rhs) {
        Self::zip_map_assign(l, r, |x, y| *x -= y);
    }

    fn mul(l: &Lhs, r: &Rhs) -> Box<Self::Output>
    where
        Self: AllocateZeros<Self::Output>,
    {
        Self::zip_map(l, r, |x, y| x * y)
    }

    fn mul_assign(l: &mut Lhs, r: &Rhs) {
        Self::zip_map_assign(l, r, |x, y| *x *= y);
    }

    fn div(l: &Lhs, r: &Rhs) -> Box<Self::Output>
    where
        Self: AllocateZeros<Self::Output>,
    {
        Self::zip_map(l, r, |x, y| x / y)
    }

    fn div_assign(l: &mut Lhs, r: &Rhs) {
        Self::zip_map_assign(l, r, |x, y| *x /= y);
    }
}

impl ZipMapElements<f32, f32> for Cpu {
    type Output = f32;
    fn zip_map_into<F: FnMut(&f32, &f32) -> f32 + Copy>(
        l: &f32,
        r: &f32,
        out: &mut Self::Output,
        mut f: F,
    ) {
        *out = f(l, r);
    }

    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(l: &mut f32, r: &f32, mut f: F) {
        f(l, r)
    }
}

impl<const M: usize> ZipMapElements<[f32; M], f32> for Cpu {
    type Output = [f32; M];
    fn zip_map_into<F: FnMut(&f32, &f32) -> f32 + Copy>(
        l: &[f32; M],
        r: &f32,
        out: &mut Self::Output,
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
    Cpu: ZipMapElements<Lhs, Rhs>,
{
    type Output = [<Self as ZipMapElements<Lhs, Rhs>>::Output; M];
    fn zip_map_into<F: FnMut(&f32, &f32) -> f32 + Copy>(
        l: &[Lhs; M],
        r: &[Rhs; M],
        out: &mut Self::Output,
        f: F,
    ) {
        for i in 0..M {
            Self::zip_map_into(&l[i], &r[i], &mut out[i], f);
        }
    }

    fn zip_map_assign<F: FnMut(&mut f32, &f32) + Copy>(l: &mut [Lhs; M], r: &[Rhs; M], f: F) {
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
        let mut a = [1.0; 3];
        let b = [2.0; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x + y);
        assert_eq!(c, [3.0; 3]);
    }

    #[test]
    fn test_1d_zip_map_broadcast_inner() {
        let mut a = [1.0; 3];
        let b = 2.0;
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x + y);
        assert_eq!(c, [3.0; 3]);
    }

    #[test]
    fn test_2d_zip_map() {
        let mut a = [[2.0; 2]; 3];
        let b = [[3.0; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x * y);
        assert_eq!(c, [[6.0; 2]; 3]);
    }

    #[test]
    fn test_2d_zip_map_broadcast_inner() {
        let mut a = [[2.0; 2]; 3];
        let b = [3.0; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x * y);
        assert_eq!(c, [[6.0; 2]; 3]);
    }

    #[test]
    fn test_3d_zip_map() {
        let mut a = [[[2.0; 5]; 2]; 3];
        let b = [[[3.0; 5]; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x * y);
        assert_eq!(c, [[[6.0; 5]; 2]; 3]);
    }

    #[test]
    fn test_3d_zip_map_broadcast_inner() {
        let mut a = [[[2.0; 5]; 2]; 3];
        let b = [[3.0; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, |x, y| x * y);
        assert_eq!(c, [[[6.0; 5]; 2]; 3]);
    }
}
