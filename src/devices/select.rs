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

macro_rules! select_01 {
    ($Axis:expr, $SrcTy:tt, $DstTy:tt, {$($Dims:tt),*}) => {
impl<$(const $Dims: usize),*> SelectAlongAxis<$SrcTy, usize, $DstTy, $Axis> for Cpu {
    fn select_axis(inp: &$SrcTy, indices: &usize, out: &mut $DstTy) {
        *out = inp[*indices];
    }
    fn select_add(inp: &mut $SrcTy, indices: &usize, out: &$DstTy) {
        Self::foreach_mr(&mut inp[*indices], out, &mut |a, b| *a += b);
    }
}
    };
}

macro_rules! select_0z {
    ($Axis:expr, $SrcTy:tt, $DstTy:tt, {$($Dims:tt),*}) => {
impl<$(const $Dims: usize),*> SelectAlongAxis<$SrcTy, [usize; Z], $DstTy, $Axis> for Cpu {
    fn select_axis(inp: &$SrcTy, indices: &[usize; Z], out: &mut $DstTy) {
        for z in 0..Z {
            out[z] = inp[indices[z]];
        }
    }
    fn select_add(inp: &mut $SrcTy, indices: &[usize; Z], out: &$DstTy) {
        for z in 0..Z {
            Self::foreach_mr(&mut inp[indices[z]], &out[z], &mut |a, b| *a += b);
        }
    }
}
    };
}

macro_rules! select_nz {
    ($Axis:expr, $SrcTy:tt, $IndTy:tt, $DstTy:tt, {$($Dims:tt),*}) => {
impl<$(const $Dims: usize),*> SelectAlongAxis<$SrcTy, $IndTy, $DstTy, $Axis> for Cpu {
    fn select_axis(inp: &$SrcTy, indices: &$IndTy, out: &mut $DstTy) {
        for m in 0..M {
            Self::select_axis(&inp[m], &indices[m], &mut out[m]);
        }
    }
    fn select_add(inp: &mut $SrcTy, indices: &$IndTy, out: &$DstTy) {
        for m in 0..M {
            Self::select_add(&mut inp[m], &indices[m], &out[m]);
        }
    }
}
    };
}

// 1d
select_01!(-1, [f32; M], f32, { M });
select_0z!(-1, [f32; M], [f32; Z], {M, Z});

// 2d
select_01!(0, [[f32; N]; M], [f32; N], {M, N});
select_0z!(0, [[f32; N]; M], [[f32; N]; Z], {M, N, Z});
select_nz!(-1, [[f32; N]; M], [usize; M], [f32; M], {M, N});
select_nz!(-1, [[f32; N]; M], [[usize; Z]; M], [[f32; Z]; M], {M, N, Z});

// 3d
select_01!(0, [[[f32; O]; N]; M], [[f32; O]; N], {M, N, O});
select_0z!(0, [[[f32; O]; N]; M], [[[f32; O]; N]; Z], {M, N, O, Z});
select_nz!(1, [[[f32; O]; N]; M], [usize; M], [[f32; O]; M], {M, N, O});
select_nz!(1, [[[f32; O]; N]; M], [[usize; Z]; M], [[[f32; O]; Z]; M], {M, N, O, Z});
select_nz!(-1, [[[f32; O]; N]; M], [[usize; N]; M], [[f32; N]; M], {M, N, O});
select_nz!(-1, [[[f32; O]; N]; M], [[[usize; Z]; N]; M], [[[f32; Z]; N]; M], {M, N, O, Z});

// 4d
select_01!(0, [[[[f32; P]; O]; N]; M], [[[f32; P]; O]; N], {M, N, O, P});
select_0z!(0, [[[[f32; P]; O]; N]; M], [[[[f32; P]; O]; N]; Z], {M, N, O, P, Z});
select_nz!(1, [[[[f32; P]; O]; N]; M], [usize; M], [[[f32; P]; O]; M], {M, N, O, P});
select_nz!(1, [[[[f32; P]; O]; N]; M], [[usize; Z]; M], [[[[f32; P]; O]; Z]; M], {M, N, O, P, Z});
select_nz!(2, [[[[f32; P]; O]; N]; M], [[usize; N]; M], [[[f32; P]; N]; M], {M, N, O, P});
select_nz!(2, [[[[f32; P]; O]; N]; M], [[[usize; Z]; N]; M], [[[[f32; P]; Z]; N]; M], {M, N, O, P, Z});
select_nz!(-1, [[[[f32; P]; O]; N]; M], [[[usize; O]; N]; M], [[[f32; O]; N]; M], {M, N, O, P});
select_nz!(-1, [[[[f32; P]; O]; N]; M], [[[[usize; Z]; O]; N]; M], [[[[f32; Z]; O]; N]; M], {M, N, O, P, Z});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;

    #[test]
    fn test_select_1d_0() {
        let a = [1.0, 2.0, 3.0];
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &1, &mut b);
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
        <Cpu as SelectAlongAxis<_, _, _, -1>>::select_axis(&a, &[0, 1], &mut b);
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
        <Cpu as SelectAlongAxis<_, _, _, -1>>::select_axis(
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
        <Cpu as SelectAlongAxis<_, _, _, -1>>::select_axis(
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
