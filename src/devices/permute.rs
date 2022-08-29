use super::Cpu;

pub trait DevicePermute2<A, B, const I: isize, const J: isize> {
    fn permute(a: &A, b: &mut B);
    fn inverse_permute(a: &mut A, b: &B);
}

pub trait DevicePermute3<A, B, const I: isize, const J: isize, const K: isize> {
    fn permute(a: &A, b: &mut B);
    fn inverse_permute(a: &mut A, b: &B);
}

pub trait DevicePermute4<A, B, const I: isize, const J: isize, const K: isize, const L: isize> {
    fn permute(a: &A, b: &mut B);
    fn inverse_permute(a: &mut A, b: &B);
}

#[rustfmt::skip]
macro_rules! axis { (0) => { M }; (1) => { N }; (2) => { O }; (3) => { P }; }

#[rustfmt::skip]
macro_rules! array {
    ($Ax0:tt) => { [f32; axis!($Ax0)] };
    ($Ax0:tt, $Ax1:tt) => { [[f32; axis!($Ax1)]; axis!($Ax0)] };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => { [[[f32; axis!($Ax2)]; axis!($Ax1)]; axis!($Ax0)] };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => { [[[[f32; axis!($Ax3)]; axis!($Ax2)]; axis!($Ax1)]; axis!($Ax0)] };
}

#[rustfmt::skip]
macro_rules! impl_permute {
    ($Ax0:tt, $Ax1:tt) => {
impl<const M: usize, const N: usize>
    DevicePermute2<array!(0, 1), array!($Ax0, $Ax1), $Ax0, $Ax1> for Cpu
{
    fn permute(a: &array!(0, 1), b: &mut array!($Ax0, $Ax1)) {
        loop2::<M, N, $Ax0, $Ax1, _>(&mut |[m, n], [i, j]| {
            b[i][j] = a[m][n];
        });
    }
    fn inverse_permute(a: &mut array!(0, 1), b: &array!($Ax0, $Ax1)) {
        loop2::<M, N, $Ax0, $Ax1, _>(&mut |[m, n], [i, j]| {
            a[m][n] = b[i][j];
        });
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt) => {
impl<const M: usize, const N: usize, const O: usize>
    DevicePermute3<array!(0, 1, 2), array!($Ax0, $Ax1, $Ax2), $Ax0, $Ax1, $Ax2> for Cpu
{
    fn permute(a: &array!(0, 1, 2), b: &mut array!($Ax0, $Ax1, $Ax2)) {
        loop3::<M, N, O, $Ax0, $Ax1, $Ax2, _>(&mut |[m, n, o], [i, j, k]| {
            b[i][j][k] = a[m][n][o];
        });
    }
    fn inverse_permute(a: &mut array!(0, 1, 2), b: &array!($Ax0, $Ax1, $Ax2)) {
        loop3::<M, N, O, $Ax0, $Ax1, $Ax2, _>(&mut |[m, n, o], [i, j, k]| {
            a[m][n][o] = b[i][j][k];
        });
    }
}
    };
    ($Ax0:tt, $Ax1:tt, $Ax2:tt, $Ax3:tt) => {
impl<const M: usize, const N: usize, const O: usize, const P: usize>
    DevicePermute4<array!(0,1,2,3), array!($Ax0,$Ax1,$Ax2,$Ax3), $Ax0,$Ax1,$Ax2,$Ax3> for Cpu
{
    fn permute(a: &array!(0, 1, 2, 3), b: &mut array!($Ax0, $Ax1, $Ax2, $Ax3)) {
        loop4::<M, N, O, P, $Ax0, $Ax1, $Ax2, $Ax3, _>(
            &mut |[m, n, o, p], [i, j, k, l]| {
                b[i][j][k][l] = a[m][n][o][p];
            },
        );
    }
    fn inverse_permute(a: &mut array!(0, 1, 2, 3), b: &array!($Ax0, $Ax1, $Ax2, $Ax3)) {
        loop4::<M, N, O, P, $Ax0, $Ax1, $Ax2, $Ax3, _>(
            &mut |[m, n, o, p], [i, j, k, l]| {
                a[m][n][o][p] = b[i][j][k][l];
            },
        );
    }
}
    };
}

fn idx<const I: isize, const N: usize>(indices: &[usize; N]) -> usize {
    if I < 0 {
        indices[(N as isize - I) as usize]
    } else {
        indices[I as usize]
    }
}

fn loop2<const M: usize, const N: usize, const I: isize, const J: isize, F>(f: &mut F)
where
    F: FnMut([usize; 2], [usize; 2]),
{
    for m in 0..M {
        for n in 0..N {
            let indices = [m, n];
            let i = idx::<I, 2>(&indices);
            let j = idx::<J, 2>(&indices);
            f(indices, [i, j]);
        }
    }
}

fn loop3<
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
                let i = idx::<I, 3>(&indices);
                let j = idx::<J, 3>(&indices);
                let k = idx::<K, 3>(&indices);
                f(indices, [i, j, k]);
            }
        }
    }
}

fn loop4<
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
                    let i = idx::<I, 4>(&indices);
                    let j = idx::<J, 4>(&indices);
                    let k = idx::<K, 4>(&indices);
                    let l = idx::<L, 4>(&indices);
                    f(indices, [i, j, k, l]);
                }
            }
        }
    }
}

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
}

permutations!([0, 1]);
permutations!([0, 1, 2]);
permutations!([0, 1, 2, 3]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::FillElements;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_2d_permute() {
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut b = [[0.0; 2]; 3];
        <Cpu as DevicePermute2<_, _, 1, 0>>::permute(&a, &mut b);
        assert_eq!(b, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);

        let mut c = [[0.0; 3]; 2];
        <Cpu as DevicePermute2<_, _, 1, 0>>::inverse_permute(&mut c, &b);
        assert_eq!(a, c);
    }

    #[test]
    fn test_3d_permute() {
        let a = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]];
        let mut b = [[[0.0; 1]; 2]; 3];
        <Cpu as DevicePermute3<_, _, 2, 1, 0>>::permute(&a, &mut b);
        assert_eq!(b, [[[1.0], [4.0]], [[2.0], [5.0]], [[3.0], [6.0]]]);

        let mut c = [[[0.0; 3]; 2]; 1];
        <Cpu as DevicePermute3<_, _, 2, 1, 0>>::inverse_permute(&mut c, &b);
        assert_eq!(a, c);
    }

    #[test]
    fn test_4d_permute() {
        let mut rng = thread_rng();
        let mut a = [[[[0.0; 9]; 7]; 5]; 3];
        Cpu::fill(&mut a, &mut |v| *v = rng.gen());

        let mut b = [[[[0.0; 3]; 5]; 9]; 7];
        <Cpu as DevicePermute4<_, _, 2, 3, 1, 0>>::permute(&a, &mut b);
        assert_ne!(b, [[[[0.0; 3]; 5]; 9]; 7]);

        let mut c = [[[[0.0; 9]; 7]; 5]; 3];
        <Cpu as DevicePermute4<_, _, 2, 3, 1, 0>>::inverse_permute(&mut c, &b);

        assert_eq!(a, c);
    }
}
