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
//! Just index into input for each index and assing to output[z]
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
    ($Axis:expr, $SrcTy:tt, $IndTy:tt, $DstTy:tt, {$($Dims:tt),*}) => {
impl<$(const $Dims: usize),*> SelectAlongAxis<$SrcTy, $IndTy, $DstTy, $Axis> for Cpu {
    fn select_axis(inp: &$SrcTy, indices: &$IndTy, out: &mut $DstTy) {
        *out = inp[*indices];
    }
    fn select_add(inp: &mut $SrcTy, indices: &$IndTy, out: &$DstTy) {
        Self::foreach_mr(&mut inp[*indices], out, &mut |a, b| *a += b);
    }
}
    };
}

macro_rules! select_0z {
    ($Axis:expr, $SrcTy:tt, $IndTy:tt, $DstTy:tt, {$($Dims:tt),*}) => {
impl<$(const $Dims: usize),*> SelectAlongAxis<$SrcTy, $IndTy, $DstTy, $Axis> for Cpu {
    fn select_axis(inp: &$SrcTy, indices: &$IndTy, out: &mut $DstTy) {
        for z in 0..Z {
            out[z] = inp[indices[z]];
        }
    }
    fn select_add(inp: &mut $SrcTy, indices: &$IndTy, out: &$DstTy) {
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
select_01!(-1, [f32; M], usize, f32, { M });
select_0z!(-1, [f32; M], [usize; Z], [f32; Z], {M, Z});

// 2d
select_01!(0, [[f32; N]; M], usize, [f32; N], {M, N});
select_0z!(0, [[f32; N]; M], [usize; Z], [[f32; N]; Z], {M, N, Z});
select_nz!(-1, [[f32; N]; M], [usize; M], [f32; M], {M, N});
select_nz!(-1, [[f32; N]; M], [[usize; Z]; M], [[f32; Z]; M], {M, N, Z});

// 3d
select_01!(0, [[[f32; O]; N]; M], usize, [[f32; O]; N], {M, N, O});
select_0z!(0, [[[f32; O]; N]; M], [usize; Z], [[[f32; O]; N]; Z], {M, N, O, Z});
select_nz!(1, [[[f32; O]; N]; M], [usize; M], [[f32; O]; M], {M, N, O});
select_nz!(1, [[[f32; O]; N]; M], [[usize; Z]; M], [[[f32; O]; Z]; M], {M, N, O, Z});
select_nz!(-1, [[[f32; O]; N]; M], [[usize; N]; M], [[f32; N]; M], {M, N, O});
select_nz!(-1, [[[f32; O]; N]; M], [[[usize; Z]; N]; M], [[[f32; Z]; N]; M], {M, N, O, Z});

// 4d
select_01!(0, [[[[f32; P]; O]; N]; M], usize, [[[f32; P]; O]; N], {M, N, O, P});
select_0z!(0, [[[[f32; P]; O]; N]; M], [usize; Z], [[[[f32; P]; O]; N]; Z], {M, N, O, P, Z});
select_nz!(1, [[[[f32; P]; O]; N]; M], [usize; M], [[[f32; P]; O]; M], {M, N, O, P});
select_nz!(1, [[[[f32; P]; O]; N]; M], [[usize; Z]; M], [[[[f32; P]; O]; Z]; M], {M, N, O, P, Z});
select_nz!(2, [[[[f32; P]; O]; N]; M], [[usize; N]; M], [[[f32; P]; N]; M], {M, N, O, P});
select_nz!(2, [[[[f32; P]; O]; N]; M], [[[usize; Z]; N]; M], [[[[f32; P]; Z]; N]; M], {M, N, O, P, Z});
select_nz!(-1, [[[[f32; P]; O]; N]; M], [[[usize; O]; N]; M], [[[f32; O]; N]; M], {M, N, O, P});
select_nz!(-1, [[[[f32; P]; O]; N]; M], [[[[usize; Z]; O]; N]; M], [[[[f32; Z]; O]; N]; M], {M, N, O, P, Z});
