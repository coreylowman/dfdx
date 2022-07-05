use super::{AllocateZeros, Cpu};
use crate::arrays::{CountElements, MultiDimensional};

/// Something that can have its last dimension (inner most dimension) reduced to 1 number.
///
/// # Implementation Details
/// We need to support ReduceLastDim for both `f32`, `[f32; M]`, and `[T; M]`.
/// However, naive recursive trait implementation can't do that.
///
/// We achieve recursive trait definition by using a 2nd intermediate trait
/// [MultiDimensional] that stores `[f32; M]` and `[T; M]` reduced versions,
/// and then make sure the recursive step uses [MultiDimensional], instead
/// of just saying Self: ReduceLastDim<T>. This works because `f32`
/// does NOT implement [MultiDimensional].
pub trait ReduceLastDim<T: CountElements>: AllocateZeros {
    const LAST_DIM: usize;
    type Reduced: CountElements<Dtype = T::Dtype>;

    fn reduce_last_dim_into<F>(inp: &T, out: &mut Self::Reduced, f: &mut F)
    where
        F: FnMut(T::Dtype, T::Dtype) -> T::Dtype;

    fn reduce_last_dim<F>(inp: &T, f: &mut F) -> Box<Self::Reduced>
    where
        F: FnMut(T::Dtype, T::Dtype) -> T::Dtype,
    {
        let mut out = Self::zeros();
        Self::reduce_last_dim_into(inp, &mut out, f);
        out
    }
}

impl ReduceLastDim<f32> for Cpu {
    const LAST_DIM: usize = 1;
    type Reduced = f32;
    fn reduce_last_dim_into<F>(inp: &f32, out: &mut Self::Reduced, _f: &mut F)
    where
        F: FnMut(f32, f32) -> f32,
    {
        *out = *inp;
    }
}

impl<const M: usize> ReduceLastDim<[f32; M]> for Cpu {
    const LAST_DIM: usize = M;
    type Reduced = f32;
    fn reduce_last_dim_into<F>(inp: &[f32; M], out: &mut Self::Reduced, f: &mut F)
    where
        F: FnMut(f32, f32) -> f32,
    {
        *out = inp.iter().cloned().reduce(f).unwrap();
    }
}
impl<T: MultiDimensional, const M: usize> ReduceLastDim<[T; M]> for Cpu
where
    Self: ReduceLastDim<T, Reduced = T::Reduced>,
{
    const LAST_DIM: usize = T::LAST_DIM_SIZE;
    type Reduced = [T::Reduced; M];
    fn reduce_last_dim_into<F>(inp: &[T; M], out: &mut Self::Reduced, f: &mut F)
    where
        F: FnMut(T::Dtype, T::Dtype) -> T::Dtype,
    {
        for (inp_i, out_i) in inp.iter().zip(out.iter_mut()) {
            Self::reduce_last_dim_into(inp_i, out_i, f);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::ZeroElements;

    use super::*;

    #[test]
    fn test_reduce_inner_0d() {
        let t = 2.0;
        let mut out = ZeroElements::ZEROS;
        Cpu::reduce_last_dim_into(&t, &mut out, &mut f32::max);
        assert_eq!(out, 2.0);
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
