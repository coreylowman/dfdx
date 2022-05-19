use crate::prelude::*;
use std::ops::Mul;

pub fn matmul<const M: usize, const N: usize, const O: usize, H: TapeHolder>(
    lhs: Tensor2D<M, N, H>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor2D<M, O, H> {
    let result = Tensor2D::new_boxed({
        let mut out: Box<[[f32; O]; M]> = Cpu::zeros();
        matmat_mul_into(lhs.data(), rhs.data(), &mut out);
        out
    });

    let mut lhs_deriv: Box<[[f32; N]; O]> = Cpu::zeros();
    let mut rhs_deriv: Box<[[f32; M]; N]> = Cpu::zeros();
    transpose_into(rhs.data(), lhs_deriv.as_mut());
    transpose_into(lhs.data(), rhs_deriv.as_mut());

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    tape_holder.add_operation(move |tape| {
        let result_grad: &[[f32; O]; M] = tape.ref_gradient(&_result);

        let mut d_grad_lhs: Box<[[f32; N]; M]> = Cpu::zeros();
        matmat_mul_into(result_grad, lhs_deriv.as_ref(), d_grad_lhs.as_mut());

        let mut d_grad_rhs: Box<[[f32; O]; N]> = Cpu::zeros();
        matmat_mul_into(rhs_deriv.as_ref(), result_grad, d_grad_rhs.as_mut());

        Cpu::add_assign(tape.mut_gradient(&lhs), d_grad_lhs.as_ref());
        Cpu::add_assign(tape.mut_gradient(&_rhs), d_grad_rhs.as_ref());
    });

    result.with_tape_holder(tape_holder)
}

pub fn vecmat_mul<const N: usize, const O: usize, H: TapeHolder>(
    lhs: Tensor1D<N, H>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor1D<O, H> {
    let result = Tensor1D::new_boxed({
        let mut out: Box<[f32; O]> = Cpu::zeros();
        vecmat_mul_into(lhs.data(), rhs.data(), out.as_mut());
        out
    });

    let mut lhs_deriv: Box<[[f32; N]; O]> = Cpu::zeros();
    transpose_into(rhs.data(), lhs_deriv.as_mut());

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    tape_holder.add_operation(move |tape| {
        let result_grad: &[f32; O] = tape.ref_gradient(&_result);

        let mut d_grad_lhs: Box<[f32; N]> = Cpu::zeros();
        vecmat_mul_into(result_grad, lhs_deriv.as_ref(), d_grad_lhs.as_mut());

        let mut d_grad_rhs: Box<[[f32; O]; N]> = Cpu::zeros();
        vecvec_mul_into(lhs.data(), result_grad, d_grad_rhs.as_mut());

        Cpu::add_assign(tape.mut_gradient(&lhs), d_grad_lhs.as_ref());
        Cpu::add_assign(tape.mut_gradient(&_rhs), d_grad_rhs.as_ref());
    });

    result.with_tape_holder(tape_holder)
}

impl<const M: usize, const N: usize, const O: usize, H: TapeHolder> Mul<&Tensor2D<N, O, NoTape>>
    for Tensor2D<M, N, H>
{
    type Output = Tensor2D<M, O, H>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        matmul(self, rhs)
    }
}

impl<const N: usize, const O: usize, H: TapeHolder> Mul<&Tensor2D<N, O, NoTape>>
    for Tensor1D<N, H>
{
    type Output = Tensor1D<O, H>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        vecmat_mul(self, rhs)
    }
}

fn matmat_mul_into<const M: usize, const N: usize, const O: usize>(
    x: &[[f32; N]; M],
    y: &[[f32; O]; N],
    out: &mut [[f32; O]; M],
) {
    for m in 0..M {
        vecmat_mul_into(&x[m], y, &mut out[m]);
    }
}

fn vecmat_mul_into<const N: usize, const O: usize>(
    x: &[f32; N],
    y: &[[f32; O]; N],
    out: &mut [f32; O],
) {
    for n in 0..N {
        let x_n = &x[n];
        let y_n = &y[n];
        for o in 0..O {
            out[o] += x_n * y_n[o];
        }
    }
}
fn vecvec_mul_into<const M: usize, const N: usize>(
    x: &[f32; M],
    y: &[f32; N],
    out: &mut [[f32; N]; M],
) {
    for n in 0..N {
        for m in 0..M {
            out[m][n] = x[m] * y[n];
        }
    }
}

fn transpose_into<const M: usize, const N: usize>(x: &[[f32; N]; M], out: &mut [[f32; M]; N]) {
    for n in 0..N {
        let out_n = &mut out[n];
        for m in 0..M {
            out_n[m] = x[m][n];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let t = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut out = [[0.0; 2]; 3];
        transpose_into(&t, &mut out);
        assert_eq!(out, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0],]);
    }

    #[test]
    fn test_vecmul() {
        let x = [1.0, 2.0, 3.0];
        let y = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
        let mut out = [0.0; 2];
        vecmat_mul_into(&x, &y, &mut out);
        assert_eq!(out, [3.0, 7.0]);
    }

    #[test]
    fn test_matmul() {
        let x = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut out = [[0.0; 2]; 4];
        matmat_mul_into(&x, &y, &mut out);
        assert_eq!(
            out,
            [[22.0, 28.0], [49.0, 64.0], [76.0, 100.0], [103.0, 136.0],]
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
        let r: Tensor2D<4, 2, WithTape> = a.trace() * &b;
        assert_eq!(
            r.data(),
            &[
                [0.62960154, 0.85549736],
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
        let r: Tensor1D<2, WithTape> = a.trace() * &b;
        assert_eq!(r.data(), &[1.2614361, 1.5543157]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[0.66719997, 0.68895, 0.6823]);
    }
}
