use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;
use std::ops::*;

/// Zip two Nd arrays together, and apply a generic function to them.
pub trait ZipMapElements<Lhs: CountElements, Rhs: CountElements>: AllocateZeros {
    /// Zip `l` and `r` together, call `f` on their elements, which will mutate `l`.
    fn zip_map_assign<F: FnMut(&mut Lhs::Dtype, &Rhs::Dtype)>(l: &mut Lhs, r: &Rhs, f: &mut F);

    /// Clones `lhs` into `out` using [Clone::clone_from()], and then calls [ZipMapElements::zip_map_assign()].
    ///
    /// Note: This can also be implemented without cloning, but would require each impl to implement
    /// zip_map_into in addition to map_assign. We use the clone implementation for the sake of
    /// reducing code.
    fn zip_map_into<F>(lhs: &Lhs, rhs: &Rhs, out: &mut Lhs, f: &mut F)
    where
        F: FnMut(&Lhs::Dtype, &Rhs::Dtype) -> Lhs::Dtype,
    {
        out.clone_from(lhs);
        Self::zip_map_assign(out, rhs, &mut |l, r| *l = f(l, r));
    }

    /// Allocates using [AllocateZeros] and then calls [ZipMapElements::zip_map_into()].
    fn zip_map<F>(l: &Lhs, r: &Rhs, mut f: F) -> Box<Lhs>
    where
        F: FnMut(&Lhs::Dtype, &Rhs::Dtype) -> Lhs::Dtype,
    {
        let mut out = Self::zeros();
        Self::zip_map_into(l, r, &mut out, &mut f);
        out
    }

    fn broadcast_rhs_first<F, const M: usize>(lhs: &[Lhs; M], rhs: &Rhs, f: &mut F) -> Box<[Lhs; M]>
    where
        F: FnMut(&Lhs::Dtype, &Rhs::Dtype) -> Lhs::Dtype,
    {
        let mut out: Box<[Lhs; M]> = Self::zeros();
        for i in 0..M {
            Self::zip_map_into(&lhs[i], rhs, &mut out[i], f);
        }
        out
    }

    fn broadcast_rhs_first_assign<F, const M: usize>(lhs: &mut [Lhs; M], rhs: &Rhs, f: &mut F)
    where
        F: FnMut(&mut Lhs::Dtype, &Rhs::Dtype),
    {
        for lhs_i in lhs.iter_mut() {
            Self::zip_map_assign(lhs_i, rhs, f);
        }
    }
}

impl ZipMapElements<f32, f32> for Cpu {
    fn zip_map_assign<F: FnMut(&mut f32, &f32)>(l: &mut f32, r: &f32, f: &mut F) {
        f(l, r)
    }
}

impl<const M: usize> ZipMapElements<[f32; M], f32> for Cpu {
    fn zip_map_assign<F: FnMut(&mut f32, &f32)>(l: &mut [f32; M], r: &f32, f: &mut F) {
        for lhs_i in l.iter_mut() {
            Self::zip_map_assign(lhs_i, r, f);
        }
    }
}

impl<Rhs, Lhs, const M: usize> ZipMapElements<[Lhs; M], [Rhs; M]> for Cpu
where
    Lhs: CountElements,
    Rhs: CountElements,
    Self: ZipMapElements<Lhs, Rhs>,
{
    fn zip_map_assign<F>(l: &mut [Lhs; M], r: &[Rhs; M], f: &mut F)
    where
        F: FnMut(&mut Lhs::Dtype, &Rhs::Dtype),
    {
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
        Cpu::zip_map_into(&a, &b, &mut c, &mut |x, y| x + y);
        assert_eq!(c, 3.0);
    }

    #[test]
    fn test_1d_zip_map() {
        let a = [1.0; 3];
        let b = [2.0; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, &mut |x, y| x + y);
        assert_eq!(c, [3.0; 3]);
    }

    #[test]
    fn test_1d_zip_map_broadcast_inner() {
        let a = [1.0; 3];
        let b = 2.0;
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, &mut |x, y| x + y);
        assert_eq!(c, [3.0; 3]);
    }

    #[test]
    fn test_2d_zip_map() {
        let a = [[2.0; 2]; 3];
        let b = [[3.0; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, &mut |x, y| x * y);
        assert_eq!(c, [[6.0; 2]; 3]);
    }

    #[test]
    fn test_2d_zip_map_broadcast_inner() {
        let a = [[2.0; 2]; 3];
        let b = [3.0; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, &mut |x, y| x * y);
        assert_eq!(c, [[6.0; 2]; 3]);
    }

    #[test]
    fn test_3d_zip_map() {
        let a = [[[2.0; 5]; 2]; 3];
        let b = [[[3.0; 5]; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, &mut |x, y| x * y);
        assert_eq!(c, [[[6.0; 5]; 2]; 3]);
    }

    #[test]
    fn test_3d_zip_map_broadcast_inner() {
        let a = [[[2.0; 5]; 2]; 3];
        let b = [[3.0; 2]; 3];
        let mut c = ZeroElements::ZEROS;
        Cpu::zip_map_into(&a, &b, &mut c, &mut |x, y| x * y);
        assert_eq!(c, [[[6.0; 5]; 2]; 3]);
    }
}
