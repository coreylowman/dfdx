//! Implementations of selecting either 1 or Z elements from an axis of an nd array.
//!
//! # Implementation Details
//! There are three cases to handle:
//!
//! ## Selecting 1 element from the 0th axis
//!
//! Just index into input using the single index and assign to output.
//!
//! ## Selecting Z elements from the 0th axis
//!
//! Just index into input for each index and assing to `output[z]`
//!
//! ## Selecting either 1 or Z elements from a non-zero axis
//!
//! Then all three arrays with have the same dimension as the 0th axis.
//! Do a for loop over the 0th axis and recurse!

use super::{Cpu, ForEachElement};
use crate::arrays::CountElements;

/// Select values from `T` using `Indices` and producing `R` along a single `AXIS`.
pub trait SelectAlongAxis<T: CountElements, Indices, R: CountElements, const AXIS: isize> {
    /// Equivalent to psuedocode `out = inp[indices]`
    fn select_axis(inp: &T, indices: &Indices, out: &mut R);

    /// `inp[indices] += out`
    fn select_add(inp: &mut T, indices: &Indices, out: &R);
}

impl<T, const M: usize> SelectAlongAxis<[T; M], usize, T, 0> for Cpu
where
    Self: ForEachElement<T>,
    T: Copy + CountElements,
    T::Dtype: for<'a> std::ops::AddAssign<&'a T::Dtype>,
{
    fn select_axis(inp: &[T; M], indices: &usize, out: &mut T) {
        *out = inp[*indices];
    }
    fn select_add(inp: &mut [T; M], indices: &usize, out: &T) {
        Self::foreach_mr(&mut inp[*indices], out, &mut |a, b| *a += b);
    }
}

impl<T, const M: usize, const Z: usize> SelectAlongAxis<[T; M], [usize; Z], [T; Z], 0> for Cpu
where
    Self: ForEachElement<T>,
    T: Copy + CountElements,
    T::Dtype: for<'a> std::ops::AddAssign<&'a T::Dtype>,
{
    fn select_axis(inp: &[T; M], indices: &[usize; Z], out: &mut [T; Z]) {
        for z in 0..Z {
            out[z] = inp[indices[z]];
        }
    }
    fn select_add(inp: &mut [T; M], indices: &[usize; Z], out: &[T; Z]) {
        for z in 0..Z {
            Self::foreach_mr(&mut inp[indices[z]], &out[z], &mut |a, b| *a += b);
        }
    }
}

macro_rules! select_nz {
    ($Axis:expr, $SubAxis:expr) => {
        impl<T, I, R, const M: usize> SelectAlongAxis<[T; M], [I; M], [R; M], $Axis> for Cpu
        where
            Self: SelectAlongAxis<T, I, R, $SubAxis>,
            T: CountElements,
            R: CountElements,
        {
            fn select_axis(inp: &[T; M], indices: &[I; M], out: &mut [R; M]) {
                for m in 0..M {
                    Self::select_axis(&inp[m], &indices[m], &mut out[m]);
                }
            }
            fn select_add(inp: &mut [T; M], indices: &[I; M], out: &[R; M]) {
                for m in 0..M {
                    Self::select_add(&mut inp[m], &indices[m], &out[m]);
                }
            }
        }
    };
}

select_nz!(1, 0);
select_nz!(2, 1);
select_nz!(3, 2);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;

    #[test]
    fn test_select_1d_0() {
        let a = [1.0, 2.0, 3.0];
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &1usize, &mut b);
        assert_eq!(b, 2.0);
    }

    #[test]
    fn test_select_1d_0z() {
        let a = [1.0, 2.0, 3.0];
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &[0, 1, 2, 2, 1, 0], &mut b);
        assert_eq!(b, [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
    }

    const A_2D: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    #[test]
    fn test_select_2d_0() {
        let a = A_2D;
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &0, &mut b);
        assert_eq!(b, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_select_2d_0z() {
        let a = A_2D;
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &[0, 0, 1], &mut b);
        assert_eq!(b, [a[0], a[0], a[1]]);
    }

    #[test]
    fn test_select_2d_1() {
        let a = A_2D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as SelectAlongAxis<_, _, _, 1>>::select_axis(&a, &[0, 1], &mut b);
        assert_eq!(b, [1.0, 5.0]);
    }

    #[test]
    fn test_select_2d_1z() {
        let a = A_2D;
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &[[0, 2], [1, 1]], &mut b);
        assert_eq!(b, [[1.0, 3.0], [5.0, 5.0]]);
    }

    #[test]
    fn test_select_add_2d() {
        let mut a = [[0.0; 3]; 2];
        let b = [[1.0, 3.0], [5.0, 5.0]];
        let i = [[0, 2], [1, 1]];
        Cpu::select_add(&mut a, &i, &b);
        assert_eq!(a, [[1.0, 0.0, 3.0], [0.0, 10.0, 0.0]]);
    }

    const A_3D: [[[f32; 3]; 2]; 4] = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]],
        [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
        [[1.0, 2.0, -3.0], [-4.0, -5.0, -6.0]],
    ];

    #[test]
    fn test_select_3d_0() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &0, &mut b);
        assert_eq!(b, A_3D[0]);
    }

    #[test]
    fn test_select_3d_0z() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &[0, 0, 1, 2, 3, 3], &mut b);
        assert_eq!(b, [A_3D[0], A_3D[0], A_3D[1], A_3D[2], A_3D[3], A_3D[3]]);
    }

    #[test]
    fn test_select_3d_1() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as SelectAlongAxis<_, _, _, 1>>::select_axis(&a, &[0, 0, 1, 1], &mut b);
        assert_eq!(b, [A_3D[0][0], A_3D[1][0], A_3D[2][1], A_3D[3][1]]);
    }

    #[test]
    fn test_select_3d_1z() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as SelectAlongAxis<_, _, _, 1>>::select_axis(&a, &[[0], [0], [1], [1]], &mut b);
        assert_eq!(b, [[A_3D[0][0]], [A_3D[1][0]], [A_3D[2][1]], [A_3D[3][1]]]);
    }

    #[test]
    fn test_select_3d_2() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as SelectAlongAxis<_, _, _, 2>>::select_axis(
            &a,
            &[[1, 0], [0, 1], [0, 0], [1, 1]],
            &mut b,
        );
        assert_eq!(
            b,
            [
                [A_3D[0][0][1], A_3D[0][1][0]],
                [A_3D[1][0][0], A_3D[1][1][1]],
                [A_3D[2][0][0], A_3D[2][1][0]],
                [A_3D[3][0][1], A_3D[3][1][1]],
            ]
        );
    }

    #[test]
    fn test_select_3d_2z() {
        let a = A_3D;
        let mut b: [[[f32; 1]; 2]; 4] = ZeroElements::ZEROS;
        <Cpu as SelectAlongAxis<_, _, _, 2>>::select_axis(
            &a,
            &[[[1], [0]], [[0], [1]], [[0], [0]], [[1], [1]]],
            &mut b,
        );
        assert_eq!(
            b,
            [
                [[A_3D[0][0][1]], [A_3D[0][1][0]]],
                [[A_3D[1][0][0]], [A_3D[1][1][1]]],
                [[A_3D[2][0][0]], [A_3D[2][1][0]]],
                [[A_3D[3][0][1]], [A_3D[3][1][1]]],
            ]
        );
    }
}
