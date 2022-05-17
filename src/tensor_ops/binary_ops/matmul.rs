use crate::prelude::*;
use std::ops::Mul;

pub fn matmat_mul<const M: usize, const N: usize, const O: usize, H: TapeHolder>(
    lhs: Tensor2D<M, N, H>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor2D<M, O, H> {
    let result = Tensor2D::new(matmul_arrays(lhs.data(), rhs.data()));
    let (lhs, mut tape_holder) = lhs.split_tape_holder();

    let lhs_deriv = transpose_arrays(rhs.data());
    let rhs_deriv = transpose_arrays(lhs.data());
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let result_grad = tape.ref_gradient(&_result);
        let d_grad_lhs = matmul_arrays(result_grad, &lhs_deriv);
        let d_grad_rhs = matmul_arrays(&rhs_deriv, result_grad);

        Cpu::add_assign(tape.mut_gradient(&lhs), &d_grad_lhs);
        Cpu::add_assign(tape.mut_gradient(&_rhs), &d_grad_rhs);
    });

    result.with_tape_holder(tape_holder)
}

pub fn vecmat_mul<const N: usize, const O: usize, H: TapeHolder>(
    lhs: Tensor1D<N, H>,
    rhs: &Tensor2D<N, O, NoTape>,
) -> Tensor1D<O, H> {
    let result = Tensor1D::new(vecmul_arrays(lhs.data(), rhs.data()));

    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let lhs_deriv = transpose_arrays(rhs.data());
    let rhs_deriv = transpose_arrays(&[*lhs.data()]);
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let result_grad = tape.ref_gradient(&_result);
        let d_grad_lhs = vecmul_arrays(result_grad, &lhs_deriv);
        let d_grad_rhs = matmul_arrays(&rhs_deriv, &[*result_grad]);

        Cpu::add_assign(tape.mut_gradient(&lhs), &d_grad_lhs);
        Cpu::add_assign(tape.mut_gradient(&_rhs), &d_grad_rhs);
    });

    result.with_tape_holder(tape_holder)
}

impl<const M: usize, const N: usize, const O: usize, H: TapeHolder> Mul<&Tensor2D<N, O, NoTape>>
    for Tensor2D<M, N, H>
{
    type Output = Tensor2D<M, O, H>;
    fn mul(self, rhs: &Tensor2D<N, O, NoTape>) -> Self::Output {
        matmat_mul(self, rhs)
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

fn matmul_arrays<const M: usize, const N: usize, const O: usize>(
    x: &[[f32; N]; M],
    y: &[[f32; O]; N],
) -> [[f32; O]; M] {
    let mut result = [[0.0; O]; M];
    for m in 0..M {
        result[m] = vecmul_arrays(&x[m], y);
    }
    result
}

fn vecmul_arrays<const N: usize, const O: usize>(x: &[f32; N], y: &[[f32; O]; N]) -> [f32; O] {
    let mut result = [0.0; O];
    for n in 0..N {
        let x_n = &x[n];
        for o in 0..O {
            result[o] += x_n * &y[n][o];
        }
    }
    result
}

fn transpose_arrays<const M: usize, const N: usize>(x: &[[f32; N]; M]) -> [[f32; M]; N] {
    let mut result = [[0.0; M]; N];
    for n in 0..N {
        for m in 0..M {
            result[n][m] = x[m][n];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let t = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(transpose_arrays(&t), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0],]);
    }

    #[test]
    fn test_vecmul() {
        let x = [1.0, 2.0, 3.0];
        let y = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
        assert_eq!(vecmul_arrays(&x, &y), [3.0, 7.0]);
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
        assert_eq!(
            matmul_arrays(&x, &y),
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
