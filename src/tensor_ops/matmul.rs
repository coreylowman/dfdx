use super::utils::move_tape_and_add_backward_binop;
use crate::prelude::*;
use matrixmultiply::sgemm;

/// Matrix multiplication.
///
/// # Generics
/// - `M`: number of rows of `lhs`.
/// - `K`: number of columns of `lhs` and number of rows of `rhs`.
/// - `N`: Number of columns of `rhs`.
///
/// # Arguments
/// * `lhs` - a 2d tensor representing a MxK matrix
/// * `rhs` - a 2d tensor representing a KxN matrix
///
/// Returns a 2d tensor representing an MxN matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor2D<3, 2> = Tensor2D::zeros();
/// let y: Tensor2D<2, 4> = Tensor2D::zeros();
/// let result: Tensor2D<3, 4> = matmul(x, &y);
/// ```
pub fn matmul<const M: usize, const K: usize, const N: usize, TAPE: Tape>(
    lhs: Tensor2D<M, K, TAPE>,
    rhs: &Tensor2D<K, N, NoneTape>,
) -> Tensor2D<M, N, TAPE> {
    let mut result = Tensor2D::zeros();
    mm(lhs.data(), rhs.data(), result.mut_data());

    // copy rhs data for use later when computing gradients
    let rhs_data = rhs.data.clone();

    move_tape_and_add_backward_binop(lhs, rhs, result, move |lhs, rhs, result, grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &result);
        mm_bt(result_grad, rhs_data.as_ref(), lhs_grad);

        let (rhs_grad, result_grad) = grads.mut_and_ref(&rhs, &result);
        mm_at(lhs.data(), result_grad, rhs_grad);
    })
}

/// Matrix multiplication with the transpose of `rhs`. Equivalent to `matmul(lhs, transpose(rhs))`.
///
/// # Arguments
/// * `lhs` - a 2d tensor representing a MxK matrix
/// * `rhs_t` - a 2d tensor representing a NxK matrix.
///
/// # Generics
/// - `M`: number of rows of `lhs`.
/// - `K`: number of columns of `lhs` and number of rows of `rhs`.
/// - `N`: Number of columns of `rhs`.
///
/// Returns a 2d tensor representing an MxO matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor2D<3, 2> = Tensor2D::zeros();
/// let y: Tensor2D<4, 2> = Tensor2D::zeros();
/// let result: Tensor2D<3, 4> = matmul_transpose(x, &y);
/// ```
pub fn matmul_transpose<const M: usize, const K: usize, const N: usize, TAPE: Tape>(
    lhs: Tensor2D<M, K, TAPE>,
    rhs_t: &Tensor2D<N, K, NoneTape>,
) -> Tensor2D<M, N, TAPE> {
    let mut result = Tensor2D::zeros();
    mm_bt(lhs.data(), rhs_t.data(), result.mut_data());

    // copy rhs data for use later when computing gradients
    let rhs_data = rhs_t.data.clone();

    move_tape_and_add_backward_binop(lhs, rhs_t, result, move |lhs, rhs, result, grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &result);
        mm(result_grad, rhs_data.as_ref(), lhs_grad);

        let (rhs_t_grad, result_grad) = grads.mut_and_ref(&rhs, &result);
        mm_atct(lhs.data(), result_grad, rhs_t_grad);
    })
}

/// vector * matrix multiplication.
///
/// This is equivalent to matrix multiplication with M == 1.
///
/// # Generics
/// - `K`: number of columns of `lhs` and number of rows of `rhs`.
/// - `N`: Number of columns of `rhs`.
///
/// # Arguments
/// * `lhs` - a 1d tensor representing a 1xK matrix
/// * `rhs` - a 2d tensor representing a KxN matrix
///
/// Returns a 1d tensor representing an 1xN matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor1D<2> = Tensor1D::zeros();
/// let y: Tensor2D<2, 4> = Tensor2D::zeros();
/// let result: Tensor1D<4> = vecmat_mul(x, &y);
/// ```
pub fn vecmat_mul<const K: usize, const N: usize, TAPE: Tape>(
    lhs: Tensor1D<K, TAPE>,
    rhs: &Tensor2D<K, N, NoneTape>,
) -> Tensor1D<N, TAPE> {
    let mut result = Tensor1D::zeros();
    vm(lhs.data(), rhs.data(), result.mut_data());

    let rhs_data = rhs.data.clone();

    move_tape_and_add_backward_binop(lhs, rhs, result, move |lhs, rhs, result, grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &result);
        vm_bt(result_grad, rhs_data.as_ref(), lhs_grad);

        let (rhs_t_grad, result_grad) = grads.mut_and_ref(&rhs, &result);
        vv(lhs.data(), result_grad, rhs_t_grad);
    })
}

/// vector * matrix multiplication where `rhs` is transposed. `y * transpose(rhs)`
///
/// # Arguments
/// * `lhs` - a 1d tensor representing a 1xK matrix
/// * `rhs_t` - a 2d tensor representing a NxK matrix
///
/// # Generics
/// - `K`: number of columns of `lhs` and number of rows of `rhs`.
/// - `N`: Number of columns of `rhs`.
///
/// Returns a 1d tensor representing an 1xO matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor1D<2> = Tensor1D::zeros();
/// let y: Tensor2D<4, 2> = Tensor2D::zeros();
/// let result: Tensor1D<4> = vecmat_mul_transpose(x, &y);
/// ```
pub fn vecmat_mul_transpose<const K: usize, const N: usize, TAPE: Tape>(
    lhs: Tensor1D<K, TAPE>,
    rhs_t: &Tensor2D<N, K, NoneTape>,
) -> Tensor1D<N, TAPE> {
    let mut result = Tensor1D::zeros();
    vm_bt(lhs.data(), rhs_t.data(), result.mut_data());

    let rhs_t_data = rhs_t.data.clone();

    move_tape_and_add_backward_binop(lhs, rhs_t, result, move |lhs, rhs, result, grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &result);
        vm(result_grad, rhs_t_data.as_ref(), lhs_grad);

        let (rhs_t_grad, result_grad) = grads.mut_and_ref(&rhs, &result);
        vv(result_grad, lhs.data(), rhs_t_grad);
    })
}

/// matrix multiply `c += a * b`
fn mm<const M: usize, const K: usize, const N: usize>(
    a: &[[f32; K]; M],
    b: &[[f32; N]; K],
    c: &mut [[f32; N]; M],
) {
    let a = a.as_ptr() as *const f32;
    let b = b.as_ptr() as *const f32;
    let c = c.as_mut_ptr() as *mut f32;
    unsafe {
        sgemm(
            M, K, N, 1.0, a, K as isize, 1, b, N as isize, 1, 1.0, c, N as isize, 1,
        )
    }
}

/// matrix multiply `c += trans(a) * b`
fn mm_at<const M: usize, const K: usize, const N: usize>(
    a_t: &[[f32; M]; K],
    b: &[[f32; N]; K],
    c: &mut [[f32; N]; M],
) {
    let a_t = a_t.as_ptr() as *const f32;
    let b = b.as_ptr() as *const f32;
    let c = c.as_mut_ptr() as *mut f32;
    unsafe {
        sgemm(
            M, K, N, 1.0, a_t, 1, M as isize, b, N as isize, 1, 1.0, c, N as isize, 1,
        )
    }
}

/// matrix multiply `c += a * trans(b)`
fn mm_bt<const M: usize, const K: usize, const N: usize>(
    a: &[[f32; K]; M],
    b_t: &[[f32; K]; N],
    c: &mut [[f32; N]; M],
) {
    let a = a.as_ptr() as *const f32;
    let b_t = b_t.as_ptr() as *const f32;
    let c = c.as_mut_ptr() as *mut f32;
    unsafe {
        sgemm(
            M, K, N, 1.0, a, K as isize, 1, b_t, 1, K as isize, 1.0, c, N as isize, 1,
        )
    }
}

/// matrix multiply `trans(c) += trans(a) * b`
fn mm_atct<const M: usize, const K: usize, const N: usize>(
    a_t: &[[f32; M]; K],
    b: &[[f32; N]; K],
    c_t: &mut [[f32; M]; N],
) {
    let a_t = a_t.as_ptr() as *const f32;
    let b = b.as_ptr() as *const f32;
    let c_t = c_t.as_mut_ptr() as *mut f32;
    unsafe {
        sgemm(
            M, K, N, 1.0, a_t, 1, M as isize, b, N as isize, 1, 1.0, c_t, 1, M as isize,
        )
    }
}

/// vector matrix multiply `c += a * b`
fn vm<const K: usize, const N: usize>(a: &[f32; K], b: &[[f32; N]; K], c: &mut [f32; N]) {
    const M: usize = 1;
    let a = a.as_ptr();
    let b = b.as_ptr() as *const f32;
    let c = c.as_mut_ptr();
    unsafe {
        sgemm(
            M, K, N, 1.0, a, K as isize, 1, b, N as isize, 1, 1.0, c, N as isize, 1,
        )
    }
}

/// vector matrix multiply `c += a * trans(b)`
fn vm_bt<const K: usize, const N: usize>(a: &[f32; K], b_t: &[[f32; K]; N], c: &mut [f32; N]) {
    const M: usize = 1;
    let a = a.as_ptr();
    let b_t = b_t.as_ptr() as *const f32;
    let c = c.as_mut_ptr();
    unsafe {
        sgemm(
            M, K, N, 1.0, a, K as isize, 1, b_t, 1, K as isize, 1.0, c, N as isize, 1,
        )
    }
}

/// vector vector
fn vv<const M: usize, const N: usize>(a: &[f32; M], b: &[f32; N], c: &mut [[f32; N]; M]) {
    const K: usize = 1;
    let a = a.as_ptr();
    let b = b.as_ptr();
    let c = c.as_mut_ptr() as *mut f32;
    unsafe {
        sgemm(
            M, K, N, 1.0, a, K as isize, 1, b, N as isize, 1, 1.0, c, N as isize, 1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vecmul() {
        let x = [1.0, 2.0, 3.0];
        let y = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
        let y_t = [[1.0, 0.5, 1.0 / 3.0], [2.0, 1.0, 1.0]];
        let expected = [3.0, 7.0];

        let mut out = [0.0; 2];
        vm(&x, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [0.0; 2];
        vm_bt(&x, &y_t, &mut out);
        assert_eq!(out, expected);
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
        mm(&x, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [[0.0; 2]; 4];
        mm_at(&x_t, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [[0.0; 2]; 4];
        mm_bt(&x, &y_t, &mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_vecvec() {
        let x = [1.0, 2.0, 3.0];
        let y = [-1.0, 0.5, -1.0 / 3.0, 0.25];

        let mut out = [[0.0; 4]; 3];
        vv(&x, &y, &mut out);
        assert_eq!(
            out,
            [
                [-1.0, 0.5, -1.0 / 3.0, 0.25],
                [-2.0, 1.0, -2.0 / 3.0, 0.5],
                [-3.0, 1.5, -1.0, 0.75],
            ]
        );

        let mut out = [[0.0; 3]; 4];
        vv(&y, &x, &mut out);
        assert_eq!(
            out,
            [
                [-1.0, -2.0, -3.0],
                [0.5, 1.0, 1.5],
                [-1.0 / 3.0, -2.0 / 3.0, -1.0],
                [0.25, 0.5, 0.75],
            ]
        );
    }

    #[test]
    fn test_matmat_mul() {
        let a = Tensor2D::new([
            [0.5086, 0.5234, 0.2684],
            [0.8075, 0.8437, 0.9951],
            [0.0774, 0.7539, 0.8894],
            [0.8119, 0.2693, 0.7249],
        ]);
        let b = Tensor2D::new([[0.4651, 0.9106], [0.3360, 0.5534], [0.8092, 0.3827]]);
        let r: Tensor2D<4, 2, OwnedTape> = matmul(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [0.62960154, 0.8554974],
                [1.4642863, 1.5830379],
                [1.0090116, 0.82806206],
                [1.0546886, 1.165766]
            ]
        );
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.37689444, 0.24156547, 0.30238447],
                [0.80570966, 0.5184905, 0.6703743],
                [0.4199963, 0.2735345, 0.38693744],
                [0.5321113, 0.34252504, 0.4438907]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [0.8737376, 0.9888564],
                [0.9339924, 0.991189],
                [1.1659734, 1.2298465]
            ]
        );
    }

    #[test]
    fn test_matmul_transpose() {
        let a = Tensor2D::new([
            [0.5086, 0.5234, 0.2684],
            [0.8075, 0.8437, 0.9951],
            [0.0774, 0.7539, 0.8894],
            [0.8119, 0.2693, 0.7249],
        ]);
        let b = Tensor2D::new([[0.4651, 0.3360, 0.8092], [0.9106, 0.5534, 0.3827]]);
        let r: Tensor2D<4, 2, OwnedTape> = matmul_transpose(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [0.62960154, 0.8554974],
                [1.4642863, 1.5830379],
                [1.0090116, 0.82806206],
                [1.0546886, 1.165766]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[[0.1719625, 0.111175, 0.1489875]; 4]
        );
    }

    #[test]
    fn test_vecmat_mul() {
        let a = Tensor1D::new([0.7296, 0.3974, 0.9487]);
        let b = Tensor2D::new([[0.7804, 0.5540], [0.5378, 0.8401], [0.5042, 0.8604]]);
        let r: Tensor1D<2, OwnedTape> = vecmat_mul(a.trace(), &b);
        assert_eq!(r.data(), &[1.261436, 1.5543157]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[2.6883178, 2.9369607, 2.9256766]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [1.2879219, 1.7261779],
                [0.70150787, 0.94021803],
                [1.6746868, 2.244552]
            ]
        );
    }

    #[test]
    fn test_vecmat_mul_transpose() {
        let a = Tensor1D::new([0.7296, 0.3974, 0.9487]);
        let b = Tensor2D::new([[0.7804, 0.5378, 0.5042], [0.5540, 0.8401, 0.8604]]);
        let r: Tensor1D<2, OwnedTape> = vecmat_mul_transpose(a.trace(), &b);
        assert_eq!(r.data(), &[1.261436, 1.5543157]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[0.66719997, 0.68895, 0.6823]);
    }
}
