use crate::prelude::*;
use matrixmultiply::sgemm;
use std::ops::Mul;

/// Matrix multiplication.
///
/// # Arguments
/// * `lhs` - a 2d tensor representing a MxN matrix
/// * `rhs` - a 2d tensor representing a NxO matrix
///
/// Returns a 2d tensor representing an MxO matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor2D<3, 2> = Tensor2D::zeros();
/// let y: Tensor2D<2, 4> = Tensor2D::zeros();
/// let result: Tensor2D<3, 4> = matmul(x, &y);
/// ```
pub fn matmul<const M: usize, const N: usize, const O: usize, H: Tape>(
    lhs: Tensor2D<M, N, H>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor2D<M, O, H> {
    let result = Tensor2D::new_boxed({
        let mut out: Box<[[f32; O]; M]> = Cpu::zeros();
        matmat_mul_into(lhs.data(), rhs.data(), &mut out);
        out
    });

    // copy rhs data for use later when computing gradients
    let rhs_data = rhs.data.clone();

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (mut lhs, mut tape) = lhs.split_tape();
    tape.add_backward_op(move |grads| {
        let result_grad: &[[f32; O]; M] = grads.ref_gradient(&_result);

        let mut d_grad_rhs: Box<[[f32; O]; N]> = Cpu::zeros();
        matmat_mul_into_xt(lhs.data(), result_grad, d_grad_rhs.as_mut());

        // write into lhs.mut_data() instead of allocating a new array.
        // NOTE: computation of d_grad_rhs requires lhs.data(), so this needs to come after
        // d_grad_rhs is computed above
        matmat_mul_into_yt(result_grad, rhs_data.as_ref(), lhs.mut_data());

        Cpu::add_assign(grads.mut_gradient(&lhs), lhs.data());
        Cpu::add_assign(grads.mut_gradient(&_rhs), d_grad_rhs.as_ref());
    });

    result.put_tape(tape)
}

/// vector * matrix multiplication.
///
/// # Arguments
/// * `lhs` - a 1d tensor representing a 1xN matrix
/// * `rhs` - a 2d tensor representing a NxO matrix
///
/// Returns a 1d tensor representing an 1xO matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor1D<2> = Tensor1D::zeros();
/// let y: Tensor2D<2, 4> = Tensor2D::zeros();
/// let result: Tensor1D<4> = vecmat_mul(x, &y);
/// ```
pub fn vecmat_mul<const N: usize, const O: usize, H: Tape>(
    lhs: Tensor1D<N, H>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor1D<O, H> {
    let result = Tensor1D::new_boxed({
        let mut out: Box<[f32; O]> = Cpu::zeros();
        vecmat_mul_into(lhs.data(), rhs.data(), out.as_mut());
        out
    });

    let rhs_data = rhs.data.clone();

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (mut lhs, mut tape) = lhs.split_tape();
    tape.add_backward_op(move |grads| {
        let result_grad: &[f32; O] = grads.ref_gradient(&_result);

        let mut d_grad_rhs: Box<[[f32; O]; N]> = Cpu::zeros();
        vecvec_mul_into(lhs.data(), result_grad, d_grad_rhs.as_mut());

        // write into lhs.mut_data() instead of allocating a new array.
        // NOTE: computation of d_grad_rhs requires lhs.data(), so this needs to come after
        // d_grad_rhs is computed above
        vecmat_mul_into_yt(result_grad, rhs_data.as_ref(), lhs.mut_data());

        Cpu::add_assign(grads.mut_gradient(&lhs), lhs.data());
        Cpu::add_assign(grads.mut_gradient(&_rhs), d_grad_rhs.as_ref());
    });

    result.put_tape(tape)
}

impl<const M: usize, const N: usize, const O: usize, H: Tape> Mul<&Tensor2D<N, O, NoTape>>
    for Tensor2D<M, N, H>
{
    type Output = Tensor2D<M, O, H>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        matmul(self, rhs)
    }
}

impl<const N: usize, const O: usize, H: Tape> Mul<&Tensor2D<N, O, NoTape>> for Tensor1D<N, H> {
    type Output = Tensor1D<O, H>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        vecmat_mul(self, rhs)
    }
}

/// matrix multiply `x * y`
fn matmat_mul_into<const M: usize, const N: usize, const O: usize>(
    x: &[[f32; N]; M],
    y: &[[f32; O]; N],
    out: &mut [[f32; O]; M],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            M, N, O, 1.0, a, N as isize, 1, b, O as isize, 1, 0.0, c, O as isize, 1,
        )
    };
}

/// matrix multiply `transpose(x) * y`
fn matmat_mul_into_xt<const M: usize, const N: usize, const O: usize>(
    x: &[[f32; M]; N],
    y: &[[f32; O]; N],
    out: &mut [[f32; O]; M],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            M, N, O, 1.0, a, 1, M as isize, b, O as isize, 1, 0.0, c, O as isize, 1,
        )
    };
}

/// matrix multiply `x * transpose(y)`
fn matmat_mul_into_yt<const M: usize, const N: usize, const O: usize>(
    x: &[[f32; N]; M],
    y: &[[f32; N]; O],
    out: &mut [[f32; O]; M],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            M, N, O, 1.0, a, N as isize, 1, b, 1, N as isize, 0.0, c, O as isize, 1,
        )
    };
}

fn vecmat_mul_into<const N: usize, const O: usize>(
    x: &[f32; N],
    y: &[[f32; O]; N],
    out: &mut [f32; O],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            1, N, O, 1.0, a, N as isize, 1, b, O as isize, 1, 0.0, c, O as isize, 1,
        )
    };
}

fn vecmat_mul_into_yt<const N: usize, const O: usize>(
    x: &[f32; N],
    y: &[[f32; N]; O],
    out: &mut [f32; O],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            1, N, O, 1.0, a, N as isize, 1, b, 1, N as isize, 0.0, c, O as isize, 1,
        )
    };
}

fn vecvec_mul_into<const M: usize, const O: usize>(
    x: &[f32; M],
    y: &[f32; O],
    out: &mut [[f32; O]; M],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            M, 1, O, 1.0, a, 1, 1, b, O as isize, 1, 0.0, c, O as isize, 1,
        )
    };
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
        vecmat_mul_into(&x, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [0.0; 2];
        vecmat_mul_into_yt(&x, &y_t, &mut out);
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
        matmat_mul_into(&x, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [[0.0; 2]; 4];
        matmat_mul_into_xt(&x_t, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [[0.0; 2]; 4];
        matmat_mul_into_yt(&x, &y_t, &mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_vecvec() {
        let x = [1.0, 2.0, 3.0];
        let y = [-1.0, 0.5, -1.0 / 3.0, 0.25];

        let mut out = [[0.0; 4]; 3];
        vecvec_mul_into(&x, &y, &mut out);
        assert_eq!(
            out,
            [
                [-1.0, 0.5, -1.0 / 3.0, 0.25],
                [-2.0, 1.0, -2.0 / 3.0, 0.5],
                [-3.0, 1.5, -1.0, 0.75],
            ]
        );

        let mut out = [[0.0; 3]; 4];
        vecvec_mul_into(&y, &x, &mut out);
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
        let r: Tensor2D<4, 2, OwnsTape> = a.trace() * &b;
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
        let r: Tensor1D<2, OwnsTape> = a.trace() * &b;
        assert_eq!(r.data(), &[1.261436, 1.5543157]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[0.66719997, 0.68895, 0.6823]);
    }
}
