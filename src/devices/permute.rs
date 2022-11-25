//! Permutation implementation is basically just looping over
//! all the elements and reordering them.
//!
//! This implementation heavily relies on macros to expand on the
//! possible versions. In the future it may be possible to refactor
//! this to use const generics, however the attempt at this on
//! initial implementation was too verbose and hard to understand.
//!
//! - `permutations!` expands all the possible permutations of axes
//! - `impl_permute!` does the actual implementation.
//! - [permuted_loop2], [permuted_loop3], [permuted_loop4], and [const_idx]
//!   are used to do the permutations.
//!
//! [permuted_loop2] takes in a function that receives the unpermuted set of
//! indices, and the permuted set of indices. This type of function enables
//! only specifying the looping & indexing logic once. Both
//! [DevicePermute::permute] and [DevicePermute::inverse_permute] share
//! this looping logic, but only differ in what they do with the indices.

use super::Cpu;
use crate::arrays::{Axes2, Axes3, Axes4, Axes5, Axes6};

/// Permutes axes of `A` resulting in `B`.
pub trait DevicePermute<A, B, Axes> {
    fn permute(a: &A, b: &mut B);
    fn inverse_permute(a: &mut A, b: &B);
}

/// Expands to the const generic for a specific axis. This is purely convention only.
#[rustfmt::skip]
macro_rules! axis { (0) => { M }; (1) => { N }; (2) => { O }; (3) => { P }; (4) => { Q }; (5) => { R }; }

/// Expands to a array type using the axes passed in.
/// E.g. `array!(2, 0, 1)` expands to `[[[f32; N]; M]; O]`
#[rustfmt::skip]
macro_rules! array {
    ($Ax0:tt) => { [f32; axis!($Ax0)] };
    ($Ax0:tt, $Ax1:tt) => { [[f32; axis!($Ax1)]; axis!($Ax0)] };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => { [[[f32; axis!($Ax2)]; axis!($Ax1)]; axis!($Ax0)] };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => { [[[[f32; axis!($Ax3)]; axis!($Ax2)]; axis!($Ax1)]; axis!($Ax0)] };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt) => { [[[[[f32; axis!($Ax4)]; axis!($Ax3)]; axis!($Ax2)]; axis!($Ax1)]; axis!($Ax0)] };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt) => { [[[[[[f32; axis!($Ax5)]; axis!($Ax4)]; axis!($Ax3)]; axis!($Ax2)]; axis!($Ax1)]; axis!($Ax0)] };
}

/// Concrete implementations for the permute and inverse permute functions.
#[rustfmt::skip]
macro_rules! impl_permute {
    ($Ax0:tt, $Ax1:tt) => {
impl<const M: usize, const N: usize>
    DevicePermute<array!(0, 1), array!($Ax0, $Ax1), Axes2<$Ax0, $Ax1>> for Cpu
{
    fn permute(a: &array!(0, 1), b: &mut array!($Ax0, $Ax1)) {
        permuted_loop2::<M, N, $Ax0, $Ax1, _>(&mut |[m, n], [i, j]| {
            b[i][j] = a[m][n];
        });
    }
    fn inverse_permute(a: &mut array!(0, 1), b: &array!($Ax0, $Ax1)) {
        permuted_loop2::<M, N, $Ax0, $Ax1, _>(&mut |[m, n], [i, j]| {
            a[m][n] = b[i][j];
        });
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => {
impl<const M: usize, const N: usize, const O: usize>
    DevicePermute<array!(0, 1, 2), array!($Ax0, $Ax1, $Ax2), Axes3<$Ax0, $Ax1, $Ax2>> for Cpu
{
    fn permute(a: &array!(0, 1, 2), b: &mut array!($Ax0, $Ax1, $Ax2)) {
        permuted_loop3::<M, N, O, $Ax0, $Ax1, $Ax2, _>(&mut |[m, n, o], [i, j, k]| {
            b[i][j][k] = a[m][n][o];
        });
    }
    fn inverse_permute(a: &mut array!(0, 1, 2), b: &array!($Ax0, $Ax1, $Ax2)) {
        permuted_loop3::<M, N, O, $Ax0, $Ax1, $Ax2, _>(&mut |[m, n, o], [i, j, k]| {
            a[m][n][o] = b[i][j][k];
        });
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => {
impl<const M: usize, const N: usize, const O: usize, const P: usize>
    DevicePermute<array!(0,1,2,3), array!($Ax0,$Ax1,$Ax2,$Ax3), Axes4<$Ax0,$Ax1,$Ax2,$Ax3>> for Cpu
{
    fn permute(a: &array!(0, 1, 2, 3), b: &mut array!($Ax0, $Ax1, $Ax2, $Ax3)) {
        permuted_loop4::<M, N, O, P, $Ax0, $Ax1, $Ax2, $Ax3, _>(
            &mut |[m, n, o, p], [i, j, k, l]| {
                b[i][j][k][l] = a[m][n][o][p];
            },
        );
    }
    fn inverse_permute(a: &mut array!(0, 1, 2, 3), b: &array!($Ax0, $Ax1, $Ax2, $Ax3)) {
        permuted_loop4::<M, N, O, P, $Ax0, $Ax1, $Ax2, $Ax3, _>(
            &mut |[m, n, o, p], [i, j, k, l]| {
                a[m][n][o][p] = b[i][j][k][l];
            },
        );
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4: tt) => {
impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize>
    DevicePermute<array!(0,1,2,3,4), array!($Ax0,$Ax1,$Ax2,$Ax3,$Ax4), Axes5<$Ax0,$Ax1,$Ax2,$Ax3,$Ax4>> for Cpu
{
    fn permute(a: &array!(0, 1, 2, 3, 4), b: &mut array!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4)) {
        permuted_loop5::<M, N, O, P, Q, $Ax0, $Ax1, $Ax2, $Ax3, $Ax4, _>(
            &mut |[m, n, o, p, q], [i, j, k, l, e]| {
                b[i][j][k][l][e] = a[m][n][o][p][q];
            },
        );
    }
    fn inverse_permute(a: &mut array!(0, 1, 2, 3, 4), b: &array!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4)) {
        permuted_loop5::<M, N, O, P, Q, $Ax0, $Ax1, $Ax2, $Ax3, $Ax4, _>(
            &mut |[m, n, o, p, q], [i, j, k, l, e]| {
                a[m][n][o][p][q] = b[i][j][k][l][e];
            },
        );
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4: tt, $Ax5: tt) => {
impl<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize, const R: usize>
    DevicePermute<array!(0,1,2,3,4,5), array!($Ax0,$Ax1,$Ax2,$Ax3,$Ax4,$Ax5), Axes6<$Ax0,$Ax1,$Ax2,$Ax3,$Ax4,$Ax5>> for Cpu
{
    fn permute(a: &array!(0, 1, 2, 3, 4, 5), b: &mut array!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5)) {
        permuted_loop6::<M, N, O, P, Q, R, $Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5, _>(
            &mut |[m, n, o, p, q, r], [i, j, k, l, e, f]| {
                b[i][j][k][l][e][f] = a[m][n][o][p][q][r];
            },
        );
    }
    fn inverse_permute(a: &mut array!(0, 1, 2, 3, 4, 5), b: &array!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5)) {
        permuted_loop6::<M, N, O, P, Q, R, $Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5, _>(
            &mut |[m, n, o, p, q, r], [i, j, k, l, e, f]| {
                a[m][n][o][p][q][r] = b[i][j][k][l][e][f];
            },
        );
    }
}
    };
}

/// Index into `indices` using the const `I`. If `I` < 0 then use `N - I`.
fn const_idx<const I: isize, const N: usize>(indices: &[usize; N]) -> usize {
    if I < 0 {
        indices[(N as isize - I) as usize]
    } else {
        indices[I as usize]
    }
}

/// Apply a function `f` to two sets of 2d indices.
fn permuted_loop2<const M: usize, const N: usize, const I: isize, const J: isize, F>(f: &mut F)
where
    F: FnMut([usize; 2], [usize; 2]),
{
    for m in 0..M {
        for n in 0..N {
            let indices = [m, n];
            let i = const_idx::<I, 2>(&indices);
            let j = const_idx::<J, 2>(&indices);
            f(indices, [i, j]);
        }
    }
}

/// Apply a function `f` to two sets of 2d indices.
fn permuted_loop3<
    const M: usize,
    const N: usize,
    const O: usize,
    const I: isize,
    const J: isize,
    const K: isize,
    F: FnMut([usize; 3], [usize; 3]),
>(
    f: &mut F,
) {
    for m in 0..M {
        for n in 0..N {
            for o in 0..O {
                let indices = [m, n, o];
                let i = const_idx::<I, 3>(&indices);
                let j = const_idx::<J, 3>(&indices);
                let k = const_idx::<K, 3>(&indices);
                f(indices, [i, j, k]);
            }
        }
    }
}

/// Apply a function `f` to two sets of 2d indices.
fn permuted_loop4<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const I: isize,
    const J: isize,
    const K: isize,
    const L: isize,
    F: FnMut([usize; 4], [usize; 4]),
>(
    f: &mut F,
) {
    for m in 0..M {
        for n in 0..N {
            for o in 0..O {
                for p in 0..P {
                    let indices = [m, n, o, p];
                    let i = const_idx::<I, 4>(&indices);
                    let j = const_idx::<J, 4>(&indices);
                    let k = const_idx::<K, 4>(&indices);
                    let l = const_idx::<L, 4>(&indices);
                    f(indices, [i, j, k, l]);
                }
            }
        }
    }
}

/// Apply a function `f` to two sets of 2d indices.
fn permuted_loop5<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    const I: isize,
    const J: isize,
    const K: isize,
    const L: isize,
    const D: isize,
    F: FnMut([usize; 5], [usize; 5]),
>(
    f: &mut F,
) {
    for m in 0..M {
        for n in 0..N {
            for o in 0..O {
                for p in 0..P {
                    for q in 0..Q {
                        let indices = [m, n, o, p, q];
                        let i = const_idx::<I, 5>(&indices);
                        let j = const_idx::<J, 5>(&indices);
                        let k = const_idx::<K, 5>(&indices);
                        let l = const_idx::<L, 5>(&indices);
                        let d = const_idx::<D, 5>(&indices);
                        f(indices, [i, j, k, l, d]);
                    }
                }
            }
        }
    }
}

/// Apply a function `f` to two sets of 2d indices.
fn permuted_loop6<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    const R: usize,
    const I: isize,
    const J: isize,
    const K: isize,
    const L: isize,
    const D: isize,
    const E: isize,
    F: FnMut([usize; 6], [usize; 6]),
>(
    f: &mut F,
) {
    for m in 0..M {
        for n in 0..N {
            for o in 0..O {
                for p in 0..P {
                    for q in 0..Q {
                        for r in 0..R {
                            let indices = [m, n, o, p, q ,r];
                            let i = const_idx::<I, 6>(&indices);
                            let j = const_idx::<J, 6>(&indices);
                            let k = const_idx::<K, 6>(&indices);
                            let l = const_idx::<L, 6>(&indices);
                            let d = const_idx::<D, 6>(&indices);
                            let e = const_idx::<E, 6>(&indices);
                            f(indices, [i, j, k, l, d, e]);
                        }
                    }
                }
            }
        }
    }
}

/// Expand out all the possible permutations for 2-4d
macro_rules! permutations {
    ([$Ax0:tt, $Ax1:tt]) => {
        impl_permute!($Ax0, $Ax1);
        impl_permute!($Ax1, $Ax0);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2]);
        permutations!($Ax1, [$Ax0, $Ax2]);
        permutations!($Ax2, [$Ax0, $Ax1]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2);
        impl_permute!($Ax0, $Ax2, $Ax1);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3);
        impl_permute!($Ax0, $Ax1, $Ax3, $Ax2);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3, $Ax4]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3, $Ax4]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3, $Ax4]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2, $Ax4]);
        permutations!($Ax4, [$Ax0, $Ax1, $Ax2, $Ax3]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3, $Ax4]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3, $Ax4]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2, $Ax4]);
        permutations!($Ax0, $Ax4, [$Ax1, $Ax2, $Ax3]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt, $Ax4:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, [$Ax3, $Ax4]);
        permutations!($Ax0, $Ax1, $Ax3, [$Ax2, $Ax4]);
        permutations!($Ax0, $Ax1, $Ax4, [$Ax2, $Ax3]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, [$Ax3:tt, $Ax4:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4);
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax4, $Ax3);
    };

    ([$Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, [$Ax1, $Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax1, [$Ax0, $Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax2, [$Ax0, $Ax1, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax3, [$Ax0, $Ax1, $Ax2, $Ax4, $Ax5]);
        permutations!($Ax4, [$Ax0, $Ax1, $Ax2, $Ax3, $Ax5]);
        permutations!($Ax5, [$Ax0, $Ax1, $Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, [$Ax1:tt, $Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, [$Ax2, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax2, [$Ax1, $Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax3, [$Ax1, $Ax2, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax4, [$Ax1, $Ax2, $Ax3, $Ax5]);
        permutations!($Ax0, $Ax5, [$Ax1, $Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, [$Ax2:tt, $Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, [$Ax3, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax3, [$Ax2, $Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax4, [$Ax2, $Ax3, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax5, [$Ax2, $Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, [$Ax3:tt, $Ax4:tt, $Ax5:tt]) => {
        permutations!($Ax0, $Ax1, $Ax2, $Ax3, [$Ax4, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax2, $Ax4, [$Ax3, $Ax5]);
        permutations!($Ax0, $Ax1, $Ax2, $Ax5, [$Ax3, $Ax4]);
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt, [$Ax4:tt, $Ax5:tt]) => {
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax4, $Ax5);
        impl_permute!($Ax0, $Ax1, $Ax2, $Ax3, $Ax5, $Ax4);
    };
}

permutations!([0, 1]);
permutations!([0, 1, 2]);
permutations!([0, 1, 2, 3]);
permutations!([0, 1, 2, 3, 4]);
permutations!([0, 1, 2, 3, 4, 5]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::FillElements;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_2d_permute() {
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut b = [[0.0; 2]; 3];
        <Cpu as DevicePermute<_, _, Axes2<1, 0>>>::permute(&a, &mut b);
        assert_eq!(b, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);

        let mut c = [[0.0; 3]; 2];
        <Cpu as DevicePermute<_, _, Axes2<1, 0>>>::inverse_permute(&mut c, &b);
        assert_eq!(a, c);
    }

    #[test]
    fn test_3d_permute() {
        let a = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]];
        let mut b = [[[0.0; 1]; 2]; 3];
        <Cpu as DevicePermute<_, _, Axes3<2, 1, 0>>>::permute(&a, &mut b);
        assert_eq!(b, [[[1.0], [4.0]], [[2.0], [5.0]], [[3.0], [6.0]]]);

        let mut c = [[[0.0; 3]; 2]; 1];
        <Cpu as DevicePermute<_, _, Axes3<2, 1, 0>>>::inverse_permute(&mut c, &b);
        assert_eq!(a, c);
    }

    #[test]
    fn test_4d_permute() {
        let mut rng = thread_rng();
        let mut a = [[[[0.0; 9]; 7]; 5]; 3];
        Cpu::fill(&mut a, &mut |v| *v = rng.gen());

        let mut b = [[[[0.0; 3]; 5]; 9]; 7];
        <Cpu as DevicePermute<_, _, Axes4<2, 3, 1, 0>>>::permute(&a, &mut b);
        assert_ne!(b, [[[[0.0; 3]; 5]; 9]; 7]);

        let mut c = [[[[0.0; 9]; 7]; 5]; 3];
        <Cpu as DevicePermute<_, _, Axes4<2, 3, 1, 0>>>::inverse_permute(&mut c, &b);

        assert_eq!(a, c);
    }

    #[test]
    fn test_5d_permute() {
        let mut rng = thread_rng();
        let mut a = [[[[[0.0; 9]; 7]; 5]; 3]; 4];
        Cpu::fill(&mut a, &mut |v| *v = rng.gen());

        let mut b = [[[[[0.0; 9]; 4]; 3]; 7]; 5];
        <Cpu as DevicePermute<_, _, Axes5<2, 3, 1, 0, 4>>>::permute(&a, &mut b);
        assert_ne!(b, [[[[[0.0; 9]; 4]; 3]; 7]; 5]);

        let mut c = [[[[[0.0; 9]; 7]; 5]; 3]; 4];
        <Cpu as DevicePermute<_, _, Axes5<2, 3, 1, 0, 4>>>::inverse_permute(&mut c, &b);

        assert_eq!(a, c);
    }

    #[test]
    fn test_6d_permute() {
        let mut rng = thread_rng();
        let mut a = [[[[[[0.0; 9]; 7]; 5]; 3]; 4]; 2];
        Cpu::fill(&mut a, &mut |v| *v = rng.gen());

        let mut b = [[[[[[0.0; 9]; 7]; 2]; 4]; 5]; 3];
        <Cpu as DevicePermute<_, _, Axes6<2, 3, 1, 0, 4, 5>>>::permute(&a, &mut b);
        assert_ne!(b, [[[[[[0.0; 9]; 7]; 2]; 4]; 5]; 3]);

        let mut c = [[[[[[0.0; 9]; 7]; 5]; 3]; 4]; 2];
        <Cpu as DevicePermute<_, _, Axes6<2, 3, 1, 0, 4, 5>>>::inverse_permute(&mut c, &b);

        assert_eq!(a, c);
    }
}
