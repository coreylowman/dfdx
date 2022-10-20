use super::Cpu;

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, cblas_sgemv as sgemv, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

pub trait Transpose {
    type T: Transpose<T = Self>;
}

impl<const M: usize, const N: usize> Transpose for [[f32; N]; M] {
    type T = [[f32; M]; N];
}

impl<Inner: Transpose, const B: usize> Transpose for [Inner; B] {
    type T = [Inner::T; B];
}

pub trait MatMul<A: Transpose, B: Transpose, C: Transpose> {
    fn mm(a: &A, b: &B, c: &mut C);
    fn mm_at(a: &A::T, b: &B, c: &mut C);
    fn mm_bt(a: &A, b: &B::T, c: &mut C);
    fn mm_atct(a: &A::T, b: &B, c: &mut C::T);
}

impl<const M: usize, const K: usize, const N: usize>
    MatMul<[[f32; K]; M], [[f32; N]; K], [[f32; N]; M]> for Cpu
{
    /// Matmul
    fn mm(a: &[[f32; K]; M], b: &[[f32; N]; K], c: &mut [[f32; N]; M]) {
        let a = a.as_ptr() as *const f32;
        let b = b.as_ptr() as *const f32;
        let c = c.as_mut_ptr() as *mut f32;

        #[cfg(not(feature = "cblas"))]
        unsafe {
            matrixmultiply::sgemm(
                M, K, N, 1.0, a, K as isize, 1, b, N as isize, 1, 1.0, c, N as isize, 1,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
            sgemm(RowMajor, NoTr, NoTr, m, n, k, 1.0, a, k, b, n, 1.0, c, n)
        }
    }

    /// Matmul, a is transposed.
    fn mm_at(a: &[[f32; M]; K], b: &[[f32; N]; K], c: &mut [[f32; N]; M]) {
        let a = a.as_ptr() as *const f32;
        let b = b.as_ptr() as *const f32;
        let c = c.as_mut_ptr() as *mut f32;

        #[cfg(not(feature = "cblas"))]
        unsafe {
            matrixmultiply::sgemm(
                M, K, N, 1.0, a, 1, M as isize, b, N as isize, 1, 1.0, c, N as isize, 1,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
            sgemm(RowMajor, Tr, NoTr, m, n, k, 1.0, a, m, b, n, 1.0, c, n)
        }
    }

    /// Matmul, b is transposed
    fn mm_bt(a: &[[f32; K]; M], b: &[[f32; K]; N], c: &mut [[f32; N]; M]) {
        let a = a.as_ptr() as *const f32;
        let b = b.as_ptr() as *const f32;
        let c = c.as_mut_ptr() as *mut f32;

        #[cfg(not(feature = "cblas"))]
        unsafe {
            matrixmultiply::sgemm(
                M, K, N, 1.0, a, K as isize, 1, b, 1, K as isize, 1.0, c, N as isize, 1,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
            sgemm(RowMajor, NoTr, Tr, m, n, k, 1.0, a, k, b, k, 1.0, c, n)
        }
    }

    /// Matmul, a and c are transposed
    fn mm_atct(a: &[[f32; M]; K], b: &[[f32; N]; K], c: &mut [[f32; M]; N]) {
        let a = a.as_ptr() as *const f32;
        let b = b.as_ptr() as *const f32;
        let c = c.as_mut_ptr() as *mut f32;

        #[cfg(not(feature = "cblas"))]
        unsafe {
            matrixmultiply::sgemm(
                M, K, N, 1.0, a, 1, M as isize, b, N as isize, 1, 1.0, c, 1, M as isize,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
            sgemm(ColMajor, NoTr, Tr, m, n, k, 1.0, a, m, b, n, 1.0, c, m)
        }
    }
}

impl<const BATCH: usize, const M: usize, const K: usize, const N: usize>
    MatMul<[[[f32; K]; M]; BATCH], [[f32; N]; K], [[[f32; N]; M]; BATCH]> for Cpu
where
    Self: MatMul<[[f32; K]; M], [[f32; N]; K], [[f32; N]; M]>,
{
    /// Broadcast `b` `BATCH` times.
    fn mm(a: &[[[f32; K]; M]; BATCH], b: &[[f32; N]; K], c: &mut [[[f32; N]; M]; BATCH]) {
        for i in 0..BATCH {
            Self::mm(&a[i], b, &mut c[i]);
        }
    }

    /// Broadcast `b` `BATCH` times.
    fn mm_at(a: &[[[f32; M]; K]; BATCH], b: &[[f32; N]; K], c: &mut [[[f32; N]; M]; BATCH]) {
        for i in 0..BATCH {
            Self::mm_at(&a[i], b, &mut c[i]);
        }
    }

    /// Broadcast `b` `BATCH` times.
    fn mm_bt(a: &[[[f32; K]; M]; BATCH], b: &[[f32; K]; N], c: &mut [[[f32; N]; M]; BATCH]) {
        for i in 0..BATCH {
            Self::mm_bt(&a[i], b, &mut c[i]);
        }
    }

    /// Broadcast `b` `BATCH` times.
    fn mm_atct(a: &[[[f32; M]; K]; BATCH], b: &[[f32; N]; K], c: &mut [[[f32; M]; N]; BATCH]) {
        for i in 0..BATCH {
            Self::mm_atct(&a[i], b, &mut c[i]);
        }
    }
}

impl<const BATCH: usize, const M: usize, const K: usize, const N: usize>
    MatMul<[[[f32; K]; M]; BATCH], [[[f32; N]; K]; BATCH], [[f32; N]; M]> for Cpu
where
    Self: MatMul<[[f32; K]; M], [[f32; N]; K], [[f32; N]; M]>,
{
    /// Broadcast `c` `BATCH` times.
    fn mm(a: &[[[f32; K]; M]; BATCH], b: &[[[f32; N]; K]; BATCH], c: &mut [[f32; N]; M]) {
        for i in 0..BATCH {
            Self::mm(&a[i], &b[i], c);
        }
    }

    /// Broadcast `c` `BATCH` times.
    fn mm_at(a: &[[[f32; M]; K]; BATCH], b: &[[[f32; N]; K]; BATCH], c: &mut [[f32; N]; M]) {
        for i in 0..BATCH {
            Self::mm_at(&a[i], &b[i], c);
        }
    }

    /// Broadcast `c` `BATCH` times.
    fn mm_bt(a: &[[[f32; K]; M]; BATCH], b: &[[[f32; K]; N]; BATCH], c: &mut [[f32; N]; M]) {
        for i in 0..BATCH {
            Self::mm_bt(&a[i], &b[i], c);
        }
    }

    /// Broadcast `c` `BATCH` times.
    fn mm_atct(a: &[[[f32; M]; K]; BATCH], b: &[[[f32; N]; K]; BATCH], c: &mut [[f32; M]; N]) {
        for i in 0..BATCH {
            Self::mm_atct(&a[i], &b[i], c);
        }
    }
}

impl<const BATCH: usize, A, B, C> MatMul<[A; BATCH], [B; BATCH], [C; BATCH]> for Cpu
where
    Self: MatMul<A, B, C>,
    A: Transpose,
    B: Transpose,
    C: Transpose,
    [A; BATCH]: Transpose<T = [A::T; BATCH]>,
    [B; BATCH]: Transpose<T = [B::T; BATCH]>,
    [C; BATCH]: Transpose<T = [C::T; BATCH]>,
{
    /// Batched matmul
    fn mm(a: &[A; BATCH], b: &[B; BATCH], c: &mut [C; BATCH]) {
        for i in 0..BATCH {
            Self::mm(&a[i], &b[i], &mut c[i]);
        }
    }

    /// Batched matmul
    fn mm_at(a: &[A::T; BATCH], b: &[B; BATCH], c: &mut [C; BATCH]) {
        for i in 0..BATCH {
            Self::mm_at(&a[i], &b[i], &mut c[i]);
        }
    }

    /// Batched matmul
    fn mm_bt(a: &[A; BATCH], b: &[B::T; BATCH], c: &mut [C; BATCH]) {
        for i in 0..BATCH {
            Self::mm_bt(&a[i], &b[i], &mut c[i]);
        }
    }

    /// Batched matmul
    fn mm_atct(a: &[A::T; BATCH], b: &[B; BATCH], c: &mut [C::T; BATCH]) {
        for i in 0..BATCH {
            Self::mm_atct(&a[i], &b[i], &mut c[i]);
        }
    }
}

pub trait MatMulOp<A: Transpose, B: Transpose, C: Transpose>:
    MatMul<A, B, C> + MatMul<C, B::T, A> + MatMul<A::T, C, B>
{
}

impl<A: Transpose, B: Transpose, C: Transpose> MatMulOp<A, B, C> for Cpu where
    Self: MatMul<A, B, C> + MatMul<C, B::T, A> + MatMul<A::T, C, B>
{
}

impl Cpu {
    /// vector matrix multiply `c += a * b`
    pub fn vm<const K: usize, const N: usize>(a: &[f32; K], b: &[[f32; N]; K], c: &mut [f32; N]) {
        let a = a.as_ptr();
        let b = b.as_ptr() as *const f32;
        let c = c.as_mut_ptr();

        #[cfg(not(feature = "cblas"))]
        unsafe {
            const M: usize = 1;
            matrixmultiply::sgemm(
                M, K, N, 1.0, a, K as isize, 1, b, N as isize, 1, 1.0, c, N as isize, 1,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (n, k) = (N as libc::c_int, K as libc::c_int);
            sgemv(ColMajor, NoTr, n, k, 1.0, b, n, a, 1, 1.0, c, 1)
        }
    }

    /// vector matrix multiply `c += a * trans(b)`
    pub fn vm_bt<const K: usize, const N: usize>(
        a: &[f32; K],
        b_t: &[[f32; K]; N],
        c: &mut [f32; N],
    ) {
        let a = a.as_ptr();
        let b_t = b_t.as_ptr() as *const f32;
        let c = c.as_mut_ptr();

        #[cfg(not(feature = "cblas"))]
        unsafe {
            const M: usize = 1;
            matrixmultiply::sgemm(
                M, K, N, 1.0, a, K as isize, 1, b_t, 1, K as isize, 1.0, c, N as isize, 1,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (n, k) = (N as libc::c_int, K as libc::c_int);
            sgemv(RowMajor, NoTr, n, k, 1.0, b_t, k, a, 1, 1.0, c, 1)
        }
    }

    /// vector vector
    pub fn vv<const M: usize, const N: usize>(a: &[f32; M], b: &[f32; N], c: &mut [[f32; N]; M]) {
        const K: usize = 1;
        let a = a.as_ptr();
        let b = b.as_ptr();
        let c = c.as_mut_ptr() as *mut f32;

        #[cfg(not(feature = "cblas"))]
        unsafe {
            matrixmultiply::sgemm(
                M, K, N, 1.0, a, K as isize, 1, b, N as isize, 1, 1.0, c, N as isize, 1,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
            sgemm(RowMajor, NoTr, NoTr, m, n, k, 1.0, a, k, b, n, 1.0, c, n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;

    #[test]
    fn test_valid_matmuls() {
        type A = [[f32; 2]; 5];
        type B = [[f32; 3]; 2];
        type C = [[f32; 3]; 5];

        // normal matmul
        let _ = <Cpu as MatMul<A, B, C>>::mm;

        // batch 3d matmul
        let _ = <Cpu as MatMul<[A; 10], [B; 10], [C; 10]>>::mm;

        // batch 4d matmul
        let _ = <Cpu as MatMul<[[A; 10]; 12], [[B; 10]; 12], [[C; 10]; 12]>>::mm;

        // broadcast matmul
        let _ = <Cpu as MatMul<[A; 10], B, [C; 10]>>::mm;
        let _ = <Cpu as MatMul<[A; 10], [B; 10], C>>::mm;

        // transposed
        let _ = <Cpu as MatMul<C, <B as Transpose>::T, A>>::mm;
        let _ = <Cpu as MatMul<<A as Transpose>::T, C, B>>::mm;
    }

    #[test]
    fn test_matmul() {
        let x = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let x_t = [
            [1.0, 4.0, 7.0, 10.0],
            [2.0, 5.0, 8.0, 11.0],
            [3.0, 6.0, 9.0, 12.0],
        ];
        let y = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_t = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
        let expected = [[22.0, 28.0], [49.0, 64.0], [76.0, 100.0], [103.0, 136.0]];

        let mut out = [[0.0; 2]; 4];
        Cpu::mm(&x, &y, &mut out);
        assert_close(&out, &expected);

        let mut out = [[0.0; 2]; 4];
        Cpu::mm_at(&x_t, &y, &mut out);
        assert_close(&out, &expected);

        let mut out = [[0.0; 2]; 4];
        Cpu::mm_bt(&x, &y_t, &mut out);
        assert_close(&out, &expected);
    }

    #[test]
    fn test_vecmul() {
        let x = [1.0, 2.0, 3.0];
        let y = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
        let y_t = [[1.0, 0.5, 1.0 / 3.0], [2.0, 1.0, 1.0]];
        let expected = [3.0, 7.0];

        let mut out = [0.0; 2];
        Cpu::vm(&x, &y, &mut out);
        assert_close(&out, &expected);

        let mut out = [0.0; 2];
        Cpu::vm_bt(&x, &y_t, &mut out);
        assert_close(&out, &expected);
    }

    #[test]
    fn test_vecvec() {
        let x = [1.0, 2.0, 3.0];
        let y = [-1.0, 0.5, -1.0 / 3.0, 0.25];

        let mut out = [[0.0; 4]; 3];
        Cpu::vv(&x, &y, &mut out);
        assert_close(
            &out,
            &[
                [-1.0, 0.5, -1.0 / 3.0, 0.25],
                [-2.0, 1.0, -2.0 / 3.0, 0.5],
                [-3.0, 1.5, -1.0, 0.75],
            ],
        );

        let mut out = [[0.0; 3]; 4];
        Cpu::vv(&y, &x, &mut out);
        assert_close(
            &out,
            &[
                [-1.0, -2.0, -3.0],
                [0.5, 1.0, 1.5],
                [-1.0 / 3.0, -2.0 / 3.0, -1.0],
                [0.25, 0.5, 0.75],
            ],
        );
    }
}
