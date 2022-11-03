//! Implementations of selecting either 1 or Z elements from an axis of an nd array.
//!
//! # Implementation Details
//! There are four cases to handle:
//!
//! ## Selecting 1 element from the 0th axis [select_modes::Index]
//!
//! Just index into input using the single index and assign to output.
//!
//! ## Selecting Z elements from the 0th axis [select_modes::Index]
//!
//! Just index into input for each index and assing to `output[z]`
//!
//! ## Selecting either 1 or Z elements from a non-zero axis [select_modes::Recurse]
//!
//! Then all three arrays with have the same dimension as the 0th axis.
//! Do a for loop over the 0th axis and recurse!
//!
//! ## Broadcasted select [select_modes::Broadcast]
//!
//! In this case only the indices & output are indexed. The input is broadcasted by
//! not indexing into it.

use super::{Cpu, ForEachElement};
use crate::arrays::CountElements;

/// Used to disambiguate trait implementations. Callees
/// must specify what kind of selection is occurring.
pub(crate) mod select_modes {
    use std::marker::PhantomData;

    /// Select the current axis.
    pub struct Index;

    /// Recurse the current axis.
    pub struct Recurse<M>(PhantomData<*const M>);

    /// Broadcast the current axis of input and recurse the indices.
    pub struct Broadcast<M>(PhantomData<*const M>);
}

use select_modes::{Broadcast, Index, Recurse};

pub(crate) type SelectAx0 = select_modes::Index;
pub(crate) type SelectAx1 = select_modes::Recurse<SelectAx0>;
pub(crate) type SelectAx2 = select_modes::Recurse<SelectAx1>;
pub(crate) type SelectAx3 = select_modes::Recurse<SelectAx2>;
pub(crate) type SelectAx4 = select_modes::Recurse<SelectAx3>;
#[cfg(tensor6d)]
pub(crate) type SelectAx5 = select_modes::Recurse<SelectAx4>;
pub(crate) type BSelectAx1 = select_modes::Broadcast<SelectAx0>;

/// Select values from `T` using indices `I`. `Mode` is used to disambiguate the impl.
pub trait DeviceSelect<T, I, Mode> {
    type Result;

    /// Equivalent to psuedocode `out = inp[indices]`
    fn select_axis(inp: &T, indices: &I, out: &mut Self::Result);

    /// `inp[indices] += out`
    fn select_add(inp: &mut T, indices: &I, out: &Self::Result);
}

// Select 1 element from 0th axis.
impl<T, const M: usize> DeviceSelect<[T; M], usize, Index> for Cpu
where
    Self: ForEachElement<T>,
    T: Copy + CountElements,
    T::Dtype: for<'a> std::ops::AddAssign<&'a T::Dtype>,
{
    type Result = T;

    fn select_axis(inp: &[T; M], indices: &usize, out: &mut Self::Result) {
        *out = inp[*indices];
    }

    fn select_add(inp: &mut [T; M], indices: &usize, out: &Self::Result) {
        Self::foreach_mr(&mut inp[*indices], out, &mut |a, b| *a += b);
    }
}

// Select Z elements from 0th axis.
impl<T, const M: usize, const Z: usize> DeviceSelect<[T; M], [usize; Z], Index> for Cpu
where
    Self: ForEachElement<T>,
    T: Copy + CountElements,
    T::Dtype: for<'a> std::ops::AddAssign<&'a T::Dtype>,
{
    type Result = [T; Z];

    fn select_axis(inp: &[T; M], indices: &[usize; Z], out: &mut Self::Result) {
        for z in 0..Z {
            out[z] = inp[indices[z]];
        }
    }
    fn select_add(inp: &mut [T; M], indices: &[usize; Z], out: &Self::Result) {
        for z in 0..Z {
            Self::foreach_mr(&mut inp[indices[z]], &out[z], &mut |a, b| *a += b);
        }
    }
}

// Select elements from non-zero axis
impl<T, I, const M: usize, SubMode> DeviceSelect<[T; M], [I; M], Recurse<SubMode>> for Cpu
where
    Self: DeviceSelect<T, I, SubMode>,
{
    type Result = [<Self as DeviceSelect<T, I, SubMode>>::Result; M];

    fn select_axis(inp: &[T; M], indices: &[I; M], out: &mut Self::Result) {
        for m in 0..M {
            Self::select_axis(&inp[m], &indices[m], &mut out[m]);
        }
    }

    fn select_add(inp: &mut [T; M], indices: &[I; M], out: &Self::Result) {
        for m in 0..M {
            Self::select_add(&mut inp[m], &indices[m], &out[m]);
        }
    }
}

// Broadcast select elements from non-zero axis.
impl<T, I, const M: usize, SubMode> DeviceSelect<T, [I; M], Broadcast<SubMode>> for Cpu
where
    Self: DeviceSelect<T, I, SubMode>,
{
    type Result = [<Self as DeviceSelect<T, I, SubMode>>::Result; M];

    fn select_axis(inp: &T, indices: &[I; M], out: &mut Self::Result) {
        for m in 0..M {
            Self::select_axis(inp, &indices[m], &mut out[m]);
        }
    }
    fn select_add(inp: &mut T, indices: &[I; M], out: &Self::Result) {
        for m in 0..M {
            Self::select_add(inp, &indices[m], &out[m]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;

    #[test]
    fn test_select_1d_0() {
        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let mut b: f32 = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &1, &mut b);
        assert_eq!(b, 2.0);
    }

    #[test]
    fn test_select_1d_0z() {
        let a: [f32; 3] = [1.0f32, 2.0, 3.0];
        let mut b: [f32; 6] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Index>>::select_axis(&a, &[0, 1, 2, 2, 1, 0], &mut b);
        assert_eq!(b, [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
    }

    const A_2D: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    #[test]
    fn test_select_2d_0() {
        let a = A_2D;
        let mut b: [f32; 3] = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &0, &mut b);
        assert_eq!(b, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_select_2d_0z() {
        let a = A_2D;
        let mut b: [[f32; 3]; 3] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Index>>::select_axis(&a, &[0, 0, 1], &mut b);
        assert_eq!(b, [a[0], a[0], a[1]]);
    }

    #[test]
    fn test_select_2d_1() {
        let a = A_2D;
        let mut b: [f32; 2] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<Index>>>::select_axis(&a, &[0, 1], &mut b);
        assert_eq!(b, [1.0, 5.0]);
    }

    #[test]
    fn test_select_2d_1z() {
        let a = A_2D;
        let mut b: [[f32; 2]; 2] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<Index>>>::select_axis(&a, &[[0, 2], [1, 1]], &mut b);
        assert_eq!(b, [[1.0, 3.0], [5.0, 5.0]]);
    }

    #[test]
    fn test_select_broadcast_2d() {
        let a = [[1.0], [2.0]];
        let i: [[usize; 3]; 4] = [[0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1]];
        let mut b: [[[f32; 1]; 3]; 4] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Broadcast<Index>>>::select_axis(&a, &i, &mut b);
        #[rustfmt::skip]
        assert_eq!(b, [[[1.], [2.], [1.]], [[2.], [2.], [2.]], [[1.], [1.], [1.]], [[2.], [1.], [2.]]]);
    }

    #[test]
    fn test_select_add_2d() {
        let mut a = [[0.0; 3]; 2];
        let b = [[1.0, 3.0], [5.0, 5.0]];
        let i = [[0, 2], [1, 1]];
        <Cpu as DeviceSelect<_, _, Recurse<Index>>>::select_add(&mut a, &i, &b);
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
        let mut b: [[f32; 3]; 2] = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &0, &mut b);
        assert_eq!(b, A_3D[0]);
    }

    #[test]
    fn test_select_3d_0z() {
        let a = A_3D;
        let mut b: [[[f32; 3]; 2]; 6] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Index>>::select_axis(&a, &[0, 0, 1, 2, 3, 3], &mut b);
        assert_eq!(b, [A_3D[0], A_3D[0], A_3D[1], A_3D[2], A_3D[3], A_3D[3]]);
    }

    #[test]
    fn test_select_3d_1() {
        let a = A_3D;
        let mut b: [[f32; 3]; 4] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<Index>>>::select_axis(&a, &[0, 0, 1, 1], &mut b);
        assert_eq!(b, [A_3D[0][0], A_3D[1][0], A_3D[2][1], A_3D[3][1]]);
    }

    #[test]
    fn test_select_3d_1z() {
        let a = A_3D;
        let mut b: [[[f32; 3]; 1]; 4] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<Index>>>::select_axis(&a, &[[0], [0], [1], [1]], &mut b);
        assert_eq!(b, [[A_3D[0][0]], [A_3D[1][0]], [A_3D[2][1]], [A_3D[3][1]]]);
    }

    #[test]
    fn test_select_3d_2() {
        let a = A_3D;
        let mut b: [[f32; 2]; 4] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<Recurse<Index>>>>::select_axis(
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
        <Cpu as DeviceSelect<_, _, Recurse<Recurse<Index>>>>::select_axis(
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
