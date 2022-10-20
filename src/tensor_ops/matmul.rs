use super::utils::move_tape_and_add_backward_binop;
use crate::devices::{Cpu, MatMul, MatMulOp, Transpose};
use crate::gradients::{Merge, Tape};
use crate::prelude::*;

/// Matrix multiplication. This also supports batched matrix multiplication,
/// and broadcasted matrix multiplication.
///
/// # Examples
/// 1. Normal matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor2D<3, 2> = TensorCreator::zeros();
/// let y: Tensor2D<2, 4> = TensorCreator::zeros();
/// let result: Tensor2D<3, 4> = matmul(x, &y);
/// ```
///
/// 2. Batched matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor3D<10, 3, 2> = TensorCreator::zeros();
/// let y: Tensor3D<10, 2, 4> = TensorCreator::zeros();
/// let result: Tensor3D<10, 3, 4> = matmul(x, &y);
/// ```
///
/// 3. Broadcasted matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor3D<10, 3, 2> = TensorCreator::zeros();
/// let y: Tensor2D<2, 4> = TensorCreator::zeros();
/// let result: Tensor3D<10, 3, 4> = matmul(x, &y);
/// ```
pub fn matmul<A, B, C>(a: A, b: B) -> <A as MatMulTyping<B>>::C
where
    A: Tensor<Dtype = f32> + MatMulTyping<B, C = C>,
    B: Tensor<Dtype = f32>,
    C: Tensor<Dtype = f32, Tape = A::Tape>,
    A::Tape: Merge<B::Tape>,
    A::Array: Transpose,
    B::Array: Transpose,
    C::Array: Transpose,
    A::Device: MatMulOp<A::Array, B::Array, C::Array>,
{
    let mut c = C::NoTape::zeros();
    A::Device::mm(a.data(), b.data(), c.mut_data());

    move_tape_and_add_backward_binop(a, b, c, move |a, b, c, grads| {
        let (a_grad, c_grad) = grads.mut_and_ref(&a, &c);
        A::Device::mm_bt(c_grad, b.data(), a_grad);

        let (b_grad, c_grad) = grads.mut_and_ref(&b, &c);
        A::Device::mm_at(a.data(), c_grad, b_grad);
    })
}

/// Enables concrete output types for generic matmul functions. Without this
/// you'd have to specify type of output.
pub trait MatMulTyping<B> {
    type C;
}

// Normal matmul
impl<const M: usize, const N: usize, const K: usize, H> MatMulTyping<Tensor2D<K, N>>
    for Tensor2D<M, K, H>
{
    type C = Tensor2D<M, N, H>;
}

// Batched matmul
impl<const B: usize, const M: usize, const N: usize, const K: usize, H>
    MatMulTyping<Tensor3D<B, K, N>> for Tensor3D<B, M, K, H>
{
    type C = Tensor3D<B, M, N, H>;
}

// Double batched matmul
impl<const B1: usize, const B2: usize, const M: usize, const N: usize, const K: usize, H>
    MatMulTyping<Tensor4D<B1, B2, K, N>> for Tensor4D<B1, B2, M, K, H>
{
    type C = Tensor4D<B1, B2, M, N, H>;
}

// Broadcasted matmul
impl<const B: usize, const M: usize, const N: usize, const K: usize, H> MatMulTyping<Tensor2D<K, N>>
    for Tensor3D<B, M, K, H>
{
    type C = Tensor3D<B, M, N, H>;
}

/// Matrix multiplication with the transpose of `rhs`. Equivalent to `matmul(lhs, transpose(rhs))`.
/// This supports the same variants as [matmul] (broadcasted, batched, etc).
///
/// # Examples
/// 1. Normal matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor2D<3, 2> = TensorCreator::zeros();
/// let y: Tensor2D<4, 2> = TensorCreator::zeros();
/// let result: Tensor2D<3, 4> = matmul_transpose(x, &y);
/// ```
///
/// 2. Batched matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor3D<10, 3, 2> = TensorCreator::zeros();
/// let y: Tensor3D<10, 4, 2> = TensorCreator::zeros();
/// let result: Tensor3D<10, 3, 4> = matmul_transpose(x, &y);
/// ```
///
/// 3. Broadcasted matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor3D<10, 3, 2> = TensorCreator::zeros();
/// let y: Tensor2D<4, 2> = TensorCreator::zeros();
/// let result: Tensor3D<10, 3, 4> = matmul_transpose(x, &y);
/// ```
pub fn matmul_transpose<A, B, C>(a: A, b: B) -> <A as MatMulTrTyping<B>>::C
where
    A: Tensor<Dtype = f32> + MatMulTrTyping<B, C = C>,
    B: Tensor<Dtype = f32>,
    C: Tensor<Dtype = f32, Tape = A::Tape>,
    A::Tape: Merge<B::Tape>,
    A::Array: Transpose,
    B::Array: Transpose,
    C::Array: Transpose,
    A::Device: MatMulOp<A::Array, <B::Array as Transpose>::T, C::Array>,
{
    let mut c = C::NoTape::zeros();
    A::Device::mm_bt(a.data(), b.data(), c.mut_data());

    move_tape_and_add_backward_binop(a, b, c, move |a, b, c, grads| {
        let (a_grad, c_grad) = grads.mut_and_ref(&a, &c);
        A::Device::mm(c_grad, b.data(), a_grad);

        let (b_grad, c_grad) = grads.mut_and_ref(&b, &c);
        A::Device::mm_atct(a.data(), c_grad, b_grad);
    })
}

/// Enables concrete output types for generic matmul functions. Without this
/// you'd have to specify type of output.
pub trait MatMulTrTyping<B> {
    type C;
}

impl<const M: usize, const N: usize, const K: usize, H> MatMulTrTyping<Tensor2D<N, K>>
    for Tensor2D<M, K, H>
{
    type C = Tensor2D<M, N, H>;
}

impl<const B: usize, const M: usize, const N: usize, const K: usize, H>
    MatMulTrTyping<Tensor3D<B, N, K>> for Tensor3D<B, M, K, H>
{
    type C = Tensor3D<B, M, N, H>;
}

impl<const B1: usize, const B2: usize, const M: usize, const N: usize, const K: usize, H>
    MatMulTrTyping<Tensor4D<B1, B2, N, K>> for Tensor4D<B1, B2, M, K, H>
{
    type C = Tensor4D<B1, B2, M, N, H>;
}

impl<const B: usize, const M: usize, const N: usize, const K: usize, H>
    MatMulTrTyping<Tensor2D<N, K>> for Tensor3D<B, M, K, H>
{
    type C = Tensor3D<B, M, N, H>;
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
/// let x: Tensor1D<2> = TensorCreator::zeros();
/// let y: Tensor2D<2, 4> = TensorCreator::zeros();
/// let result: Tensor1D<4> = vecmat_mul(x, &y);
/// ```
pub fn vecmat_mul<const K: usize, const N: usize, LhsTape: Tape, RhsTape: Tape>(
    lhs: Tensor1D<K, LhsTape>,
    rhs: Tensor2D<K, N, RhsTape>,
) -> Tensor1D<N, LhsTape>
where
    LhsTape: Merge<RhsTape>,
{
    let mut result = Tensor1D::zeros();
    Cpu::vm(lhs.data(), rhs.data(), result.mut_data());

    move_tape_and_add_backward_binop(lhs, rhs, result, move |lhs, rhs, result, grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &result);
        Cpu::vm_bt(result_grad, rhs.data(), lhs_grad);

        let (rhs_t_grad, result_grad) = grads.mut_and_ref(&rhs, &result);
        Cpu::vv(lhs.data(), result_grad, rhs_t_grad);
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
/// let x: Tensor1D<2> = TensorCreator::zeros();
/// let y: Tensor2D<4, 2> = TensorCreator::zeros();
/// let result: Tensor1D<4> = vecmat_mul_transpose(x, &y);
/// ```
pub fn vecmat_mul_transpose<const K: usize, const N: usize, LhsTape: Tape, RhsTape: Tape>(
    lhs: Tensor1D<K, LhsTape>,
    rhs_t: Tensor2D<N, K, RhsTape>,
) -> Tensor1D<N, LhsTape>
where
    LhsTape: Merge<RhsTape>,
{
    let mut result = Tensor1D::zeros();
    Cpu::vm_bt(lhs.data(), rhs_t.data(), result.mut_data());

    move_tape_and_add_backward_binop(lhs, rhs_t, result, move |lhs, rhs_t, result, grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &result);
        Cpu::vm(result_grad, rhs_t.data(), lhs_grad);

        let (rhs_t_grad, result_grad) = grads.mut_and_ref(&rhs_t, &result);
        Cpu::vv(result_grad, lhs.data(), rhs_t_grad);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;
    use crate::{devices::Device, tests::assert_close};
    use rand::thread_rng;

    #[test]
    fn test_valid_matmuls() {
        let _: Tensor2D<5, 2> = matmul(Tensor2D::<5, 3>::zeros(), Tensor2D::<3, 2>::zeros());

        let _: Tensor3D<10, 5, 2> =
            matmul(Tensor3D::<10, 5, 3>::zeros(), Tensor2D::<3, 2>::zeros());

        let _: Tensor3D<10, 5, 2> =
            matmul(Tensor3D::<10, 5, 3>::zeros(), Tensor3D::<10, 3, 2>::zeros());

        let _: Tensor4D<20, 10, 5, 2> = matmul(
            Tensor4D::<20, 10, 5, 3>::zeros(),
            Tensor4D::<20, 10, 3, 2>::zeros(),
        );
    }

    #[test]
    fn test_matmul() {
        let a = tensor([
            [0.5086, 0.5234, 0.2684],
            [0.8075, 0.8437, 0.9951],
            [0.0774, 0.7539, 0.8894],
            [0.8119, 0.2693, 0.7249],
        ]);
        let b = tensor([[0.4651, 0.9106], [0.3360, 0.5534], [0.8092, 0.3827]]);
        let r = matmul(a.trace(), b.clone());
        assert_close(
            r.data(),
            &[
                [0.62960154, 0.8554974],
                [1.4642863, 1.5830379],
                [1.0090116, 0.82806206],
                [1.0546886, 1.165766],
            ],
        );
        let gradients = backward(r.exp().mean());
        assert_close(
            gradients.ref_gradient(&a),
            &[
                [0.37689444, 0.24156547, 0.30238447],
                [0.80570966, 0.5184905, 0.6703743],
                [0.4199963, 0.2735345, 0.38693744],
                [0.5321113, 0.34252504, 0.4438907],
            ],
        );
        assert_close(
            gradients.ref_gradient(&b),
            &[
                [0.8737376, 0.9888564],
                [0.9339924, 0.991189],
                [1.1659734, 1.2298465],
            ],
        );
    }

    #[test]
    fn test_matmul_transpose() {
        let mut rng = thread_rng();
        let a: Tensor2D<4, 3> = TensorCreator::randn(&mut rng);
        let b: Tensor2D<3, 2> = TensorCreator::randn(&mut rng);
        let c = matmul(a.trace(), b.clone());

        let b_t = Tensor2D::new(transpose(b.data()));
        let c_tr = matmul_transpose(a.trace(), b_t.clone());
        assert_close(c_tr.data(), c.data());

        let gs = backward(c.exp().mean());
        let gs_tr = backward(c_tr.exp().mean());
        assert_close(gs_tr.ref_gradient(&a), gs.ref_gradient(&a));
        assert_close(gs_tr.ref_gradient(&b_t), &transpose(gs.ref_gradient(&b)));
    }

    #[test]
    fn test_broadcasted_matmul() {
        const N: usize = 5;
        let mut rng = thread_rng();
        let a: Tensor3D<N, 4, 3> = TensorCreator::randn(&mut rng);
        let b: Tensor2D<3, 2> = TensorCreator::randn(&mut rng);
        let r = matmul(a.trace(), b.clone());
        for i in 0..N {
            let sub_a = Tensor2D::new(a.data()[i]);
            assert_close(&r.data()[i], matmul(sub_a, b.clone()).data());
        }
        let gs = backward(r.sum());
        let mut sub_bs_summed = [[0.0; 2]; 3];
        for i in 0..N {
            let sub_a = Tensor2D::new(a.data()[i]);
            let sub_gs = backward(matmul(sub_a.trace(), b.clone()).sum());
            assert_close(&gs.ref_gradient(&a)[i], sub_gs.ref_gradient(&sub_a));
            <Cpu as Device<_>>::add(&mut sub_bs_summed, sub_gs.ref_gradient(&b));
        }
        assert_close(gs.ref_gradient(&b), &sub_bs_summed);
    }

    #[test]
    fn test_broadcasted_matmul_transpose() {
        let mut rng = thread_rng();
        let a: Tensor3D<2, 4, 3> = TensorCreator::randn(&mut rng);
        let b: Tensor2D<3, 2> = TensorCreator::randn(&mut rng);
        let c = matmul(a.trace(), b.clone());

        let b_t = Tensor2D::new(transpose(b.data()));
        let c_tr = matmul_transpose(a.trace(), b_t.clone());
        assert_close(c_tr.data(), c.data());

        let gs = backward(c.exp().mean());
        let gs_tr = backward(c_tr.exp().mean());
        assert_close(gs_tr.ref_gradient(&a), gs.ref_gradient(&a));
        assert_close(gs_tr.ref_gradient(&b_t), &transpose(gs.ref_gradient(&b)));
    }

    #[test]
    fn test_vecmat_mul() {
        let a = tensor([0.7296, 0.3974, 0.9487]);
        let b = tensor([[0.7804, 0.5540], [0.5378, 0.8401], [0.5042, 0.8604]]);
        let r: Tensor1D<2, OwnedTape> = vecmat_mul(a.trace(), b.clone());
        assert_close(r.data(), &[1.261436, 1.5543157]);
        let g = backward(r.exp().mean());
        assert_close(g.ref_gradient(&a), &[2.6883178, 2.9369607, 2.9256766]);
        assert_close(
            g.ref_gradient(&b),
            &[
                [1.2879219, 1.7261779],
                [0.70150787, 0.94021803],
                [1.6746868, 2.244552],
            ],
        );
    }

    #[test]
    fn test_vecmat_mul_transpose() {
        let a = tensor([0.7296, 0.3974, 0.9487]);
        let b = tensor([[0.7804, 0.5378, 0.5042], [0.5540, 0.8401, 0.8604]]);
        let r: Tensor1D<2, OwnedTape> = vecmat_mul_transpose(a.trace(), b.clone());
        assert_close(r.data(), &[1.261436, 1.5543157]);
        let g = backward(r.exp().mean());
        assert_close(g.ref_gradient(&a), &[2.6883178, 2.9369607, 2.9256766]);
        assert_close(
            g.ref_gradient(&b),
            &[
                [1.2879219, 0.70150787, 1.6746868],
                [1.7261779, 0.94021803, 2.244552],
            ],
        );
    }

    fn transpose<const M: usize, const N: usize>(a: &[[f32; N]; M]) -> [[f32; M]; N] {
        let mut t: [[f32; M]; N] = ZeroElements::ZEROS;
        for (m, a_m) in a.iter().enumerate() {
            for (n, a_mn) in a_m.iter().enumerate() {
                t[n][m] = *a_mn;
            }
        }
        t
    }
}
