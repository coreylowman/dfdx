use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;

/// Something that can have its last dimension (inner most dimension) reduced to 1 number.
///
/// Note: This currently cannot be implemented with recursive traits because we need
/// to support reduce `[f32; M]` to `f32` AND `f32` -> `f32`. This means it is also valid
/// to reduce `[f32; M]` to `[f32; M]` if recursive traits were used, which doesn't make sense.
pub trait ReduceLastDim<T: CountElements>: AllocateZeros {
    type Reduced: CountElements<Dtype = T::Dtype>;

    fn reduce_last_dim_into<F>(inp: &T, out: &mut Self::Reduced, f: &mut F)
    where
        F: FnMut(T::Dtype, T::Dtype) -> T::Dtype;

    fn reduce_last_dim<F>(inp: &T, mut f: F) -> Box<Self::Reduced>
    where
        F: FnMut(T::Dtype, T::Dtype) -> T::Dtype,
    {
        let mut out = Self::zeros();
        Self::reduce_last_dim_into(inp, &mut out, &mut f);
        out
    }
}

impl ReduceLastDim<f32> for Cpu {
    type Reduced = f32;
    fn reduce_last_dim_into<F>(inp: &f32, out: &mut Self::Reduced, _f: &mut F)
    where
        F: FnMut(f32, f32) -> f32,
    {
        *out = *inp;
    }
}

impl<const M: usize> ReduceLastDim<[f32; M]> for Cpu {
    type Reduced = f32;
    fn reduce_last_dim_into<F>(inp: &[f32; M], out: &mut Self::Reduced, f: &mut F)
    where
        F: FnMut(f32, f32) -> f32,
    {
        *out = inp.iter().cloned().reduce(f).unwrap();
    }
}

impl<const M: usize, const N: usize> ReduceLastDim<[[f32; N]; M]> for Cpu {
    type Reduced = [f32; M];
    fn reduce_last_dim_into<F>(inp: &[[f32; N]; M], out: &mut Self::Reduced, f: &mut F)
    where
        F: FnMut(f32, f32) -> f32,
    {
        for i in 0..M {
            Self::reduce_last_dim_into(&inp[i], &mut out[i], f);
        }
    }
}

impl<const M: usize, const N: usize, const O: usize> ReduceLastDim<[[[f32; O]; N]; M]> for Cpu {
    type Reduced = [[f32; N]; M];
    fn reduce_last_dim_into<F>(inp: &[[[f32; O]; N]; M], out: &mut Self::Reduced, f: &mut F)
    where
        F: FnMut(f32, f32) -> f32,
    {
        for i in 0..M {
            Self::reduce_last_dim_into(&inp[i], &mut out[i], f);
        }
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize>
    ReduceLastDim<[[[[f32; P]; O]; N]; M]> for Cpu
{
    type Reduced = [[[f32; O]; N]; M];
    fn reduce_last_dim_into<F>(inp: &[[[[f32; P]; O]; N]; M], out: &mut Self::Reduced, f: &mut F)
    where
        F: FnMut(f32, f32) -> f32,
    {
        for i in 0..M {
            Self::reduce_last_dim_into(&inp[i], &mut out[i], f);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::ZeroElements;

    use super::*;

    #[test]
    fn test_reduce_inner_0d() {
        let t = 3.14;
        let mut out = ZeroElements::ZEROS;
        Cpu::reduce_last_dim_into(&t, &mut out, &mut f32::max);
        assert_eq!(out, 3.14);
    }

    #[test]
    fn test_reduce_inner_1d() {
        let t = [1.0, 2.0, 3.0];
        let mut out: f32 = ZeroElements::ZEROS;
        Cpu::reduce_last_dim_into(&t, &mut out, &mut f32::max);
        assert_eq!(out, 3.0);
    }

    #[test]
    fn test_reduce_inner_2d() {
        let t = [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]];
        let mut out: [f32; 2] = ZeroElements::ZEROS;
        Cpu::reduce_last_dim_into(&t, &mut out, &mut f32::max);
        assert_eq!(out, [3.0, 6.0]);
    }

    #[test]
    fn test_reduce_inner_3d() {
        let t = [
            [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]],
            [[-1.0, -2.0, -3.0], [-6.0, -5.0, -4.0]],
        ];
        let mut out: [[f32; 2]; 2] = ZeroElements::ZEROS;
        Cpu::reduce_last_dim_into(&t, &mut out, &mut f32::max);
        assert_eq!(out, [[3.0, 6.0], [-1.0, -4.0]]);
    }
}
