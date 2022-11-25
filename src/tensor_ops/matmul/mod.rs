mod cpu_kernel;

use crate::{
    arrays::{Dtype, Shape},
    devices::{Device, HasErr},
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::ops::{try_binary_op, BinaryKernel};

mod internals {
    use super::*;
    use crate::arrays::{Dim, Rank1, Rank2, Rank3, Rank4, C};

    pub trait MatMulAlgebra<Rhs: Shape>: Shape {
        type Out: Shape;
    }

    impl<Outer: Dim, Inner: Dim> MatMulAlgebra<(Inner,)> for (Outer,) {
        type Out = (Outer, Inner);
    }

    impl<const K: usize, const N: usize> MatMulAlgebra<Rank2<K, N>> for Rank1<K> {
        type Out = Rank1<N>;
    }

    impl<M: Dim, const K: usize, const N: usize> MatMulAlgebra<Rank2<K, N>> for (M, C<K>) {
        type Out = (M, C<N>);
    }

    impl<Batch: Dim, Seq: Dim, const K: usize, const N: usize> MatMulAlgebra<Rank2<K, N>>
        for (Batch, Seq, C<K>)
    {
        type Out = (Batch, Seq, C<N>);
    }

    impl<const B: usize, const M: usize, const K: usize, const N: usize>
        MatMulAlgebra<Rank3<B, K, N>> for Rank3<B, M, K>
    {
        type Out = Rank3<B, M, N>;
    }

    impl<const B: usize, const S: usize, const M: usize, const K: usize, const N: usize>
        MatMulAlgebra<Rank4<B, S, K, N>> for Rank4<B, S, M, K>
    {
        type Out = Rank4<B, S, M, N>;
    }
}

/// Matrix * Matrix,, Vector * Matrix, and Vector * Vector multiplication.
///
/// This also supports batching and broadcasting depending on device implementations.
///
/// # Examples
/// 1. Normal matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor2D<3, 2> = TensorCreator::zeros();
/// let y: Tensor2D<2, 4> = TensorCreator::zeros();
/// let result: Tensor2D<3, 4> = matmul(x, y);
/// ```
///
/// 2. Batched matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor3D<10, 3, 2> = TensorCreator::zeros();
/// let y: Tensor3D<10, 2, 4> = TensorCreator::zeros();
/// let result: Tensor3D<10, 3, 4> = matmul(x, y);
/// ```
///
/// 3. Broadcasted matmul
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor3D<10, 3, 2> = TensorCreator::zeros();
/// let y: Tensor2D<2, 4> = TensorCreator::zeros();
/// let result: Tensor3D<10, 3, 4> = matmul(x, y);
/// ```
///
/// 4. Vector x Matrix
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor1D<2> = TensorCreator::zeros();
/// let y: Tensor2D<2, 4> = TensorCreator::zeros();
/// let result: Tensor1D<4> = vecmat_mul(x, y);
/// ```
///
/// 5. Vector x Vector
/// ```rust
/// todo!();
/// ```
pub trait TryMatMul<Rhs>: HasErr {
    type Output;
    fn matmul(self, rhs: Rhs) -> Self::Output {
        self.try_matmul(rhs).unwrap()
    }
    fn try_matmul(self, rhs: Rhs) -> Result<Self::Output, Self::Err>;
}

#[derive(Default, Copy, Clone)]
pub(super) struct MatMulKernelOp;

impl<Lhs: Shape, Rhs: Shape, E: Dtype, D: Device, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryMatMul<Tensor<Rhs, E, D, RhsTape>> for Tensor<Lhs, E, D, LhsTape>
where
    Lhs: internals::MatMulAlgebra<Rhs>,
    D: BinaryKernel<MatMulKernelOp, Lhs, Rhs, Lhs::Out, E>,
    LhsTape: Merge<RhsTape>,
{
    type Output = Tensor<Lhs::Out, E, D, LhsTape>;
    fn try_matmul(self, rhs: Tensor<Rhs, E, D, RhsTape>) -> Result<Self::Output, Self::Err> {
        try_binary_op(Default::default(), self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{AsArray, Randn, Zeros};
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_valid_matmuls() {
        let dev = build_test_device!();

        {
            let a: Tensor1D<3, _> = dev.zeros();
            let b: Tensor2D<3, 2, _> = dev.zeros();
            let _: Tensor1D<2, _> = a.matmul(b);
        }

        {
            let a: Tensor2D<5, 3, _> = dev.zeros();
            let b: Tensor2D<3, 2, _> = dev.zeros();
            let _: Tensor2D<5, 2, _> = a.matmul(b);
        }

        {
            let a: Tensor3D<10, 5, 3, _> = dev.zeros();
            let b: Tensor2D<3, 2, _> = dev.zeros();
            let _: Tensor3D<10, 5, 2, _> = a.matmul(b);
        }

        {
            let a: Tensor3D<10, 5, 3, _> = dev.zeros();
            let b: Tensor3D<10, 3, 2, _> = dev.zeros();
            let _: Tensor3D<10, 5, 2, _> = a.matmul(b);
        }

        {
            let a: Tensor4D<10, 20, 5, 3, _> = dev.zeros();
            let b: Tensor4D<10, 20, 3, 2, _> = dev.zeros();
            let _: Tensor4D<10, 20, 5, 2, _> = a.matmul(b);
        }
    }

    #[test]
    fn test_matmul_normal() {
        let dev = build_test_device!();

        let a = dev.tensor([
            [0.5086, 0.5234, 0.2684],
            [0.8075, 0.8437, 0.9951],
            [0.0774, 0.7539, 0.8894],
            [0.8119, 0.2693, 0.7249],
        ]);
        let b = dev.tensor([[0.4651, 0.9106], [0.3360, 0.5534], [0.8092, 0.3827]]);
        let r = a.trace().matmul(b.clone());
        assert_close(
            &r.as_array(),
            &[
                [0.62960154, 0.8554974],
                [1.4642863, 1.5830379],
                [1.0090116, 0.82806206],
                [1.0546886, 1.165766],
            ],
        );
        let g = r.exp().mean().backward();
        assert_close(
            &g.get(&a).as_array(),
            &[
                [0.37689444, 0.24156547, 0.30238447],
                [0.80570966, 0.5184905, 0.6703743],
                [0.4199963, 0.2735345, 0.38693744],
                [0.5321113, 0.34252504, 0.4438907],
            ],
        );
        assert_close(
            &g.get(&b).as_array(),
            &[
                [0.8737376, 0.9888564],
                [0.9339924, 0.991189],
                [1.1659734, 1.2298465],
            ],
        );
    }

    #[test]
    fn test_matmul_transpose() {
        let dev = build_test_device!();
        let a: Tensor2D<4, 3, _> = dev.randn();
        let b: Tensor2D<3, 2, _> = dev.randn();

        let c = a.trace().matmul(b.clone());
        let g1 = c.exp().mean().backward();

        let c2 = b.trace().permute().matmul(a.trace().permute());
        let g2 = c2.exp().mean().backward();

        assert_close(&g1.get(&a).as_array(), &g2.get(&a).as_array());
        assert_close(&g1.get(&b).as_array(), &g2.get(&b).as_array());
    }

    #[test]
    fn test_matul_broadcast() {
        const N: usize = 5;
        let dev = build_test_device!();
        let a: Tensor3D<N, 4, 3, _> = dev.randn();
        let a_array = a.as_array();
        let b: Tensor2D<3, 2, _> = dev.randn();
        let r = a.trace().matmul(b.clone());
        let r_array = r.as_array();
        for i in 0..N {
            let sub_a = dev.tensor(a_array[i]);
            let sub_c = sub_a.matmul(b.clone());
            assert_close(&r_array[i], &sub_c.as_array());
        }
        let gs = r.sum().backward();
        let a_grad = gs.get(&a).as_array();
        let mut sub_bs_summed = [[0.0; 2]; 3];
        for i in 0..N {
            let sub_a = dev.tensor(a_array[i]);
            let sub_gs = sub_a.trace().matmul(b.clone()).sum().backward();
            assert_close(&a_grad[i], &sub_gs.get(&sub_a).as_array());
            let sub_b_grad = sub_gs.get(&b).as_array();
            for x in 0..3 {
                for y in 0..2 {
                    sub_bs_summed[x][y] += sub_b_grad[x][y];
                }
            }
        }
        assert_close(&gs.get(&b).as_array(), &sub_bs_summed);
    }

    #[test]
    fn test_matmul_broadcast_actual() {
        const N: usize = 5;
        let dev = build_test_device!();
        let a: Tensor3D<N, 4, 3, _> = dev.randn();
        let b: Tensor2D<3, 2, _> = dev.randn();
        let b_up: Tensor3D<N, 3, 2, _, _> = b.trace().broadcast();
        let r1 = a.trace().matmul(b_up);
        let r2 = a.trace().matmul(b.clone());
        assert_eq!(r1.as_array(), r2.as_array());
        let g1 = r1.exp().mean().backward();
        let g2 = r2.exp().mean().backward();
        assert_eq!(g1.get(&a).as_array(), g2.get(&a).as_array());
        assert_eq!(g1.get(&b).as_array(), g2.get(&b).as_array());
    }

    #[test]
    fn test_matmul_batched_3d() {
        let dev = build_test_device!();

        let a: Tensor3D<5, 3, 2, _> = dev.randn();
        let a_array = a.as_array();
        let b: Tensor3D<5, 2, 4, _> = dev.randn();
        let b_array = b.as_array();
        let c = a.trace().matmul(b.clone());
        let c_array = c.as_array();
        let g = c.exp().sum().backward();

        let g_a = g.get(&a).as_array();
        let g_b = g.get(&b).as_array();

        for i in 0..5 {
            let sub_a = dev.tensor(a_array[i]);
            let sub_b = dev.tensor(b_array[i]);
            let sub_c = sub_a.trace().matmul(sub_b.clone());
            assert_eq!(sub_c.as_array(), c_array[i]);
            let sub_g = sub_c.exp().sum().backward();
            assert_eq!(sub_g.get(&sub_a).as_array(), g_a[i]);
            assert_eq!(sub_g.get(&sub_b).as_array(), g_b[i]);
        }
    }

    #[test]
    fn test_matmul_batched_4d() {
        let dev = build_test_device!();

        let a: Tensor4D<7, 5, 3, 2, _> = dev.randn();
        let a_array = a.as_array();
        let b: Tensor4D<7, 5, 2, 4, _> = dev.randn();
        let b_array = b.as_array();
        let c = a.trace().matmul(b.clone());
        let c_array = c.as_array();
        let g = c.exp().sum().backward();

        let g_a = g.get(&a).as_array();
        let g_b = g.get(&b).as_array();

        for i in 0..7 {
            for j in 0..5 {
                let sub_a = dev.tensor(a_array[i][j]);
                let sub_b = dev.tensor(b_array[i][j]);
                let sub_c = sub_a.trace().matmul(sub_b.clone());
                assert_eq!(sub_c.as_array(), c_array[i][j]);
                let sub_g = sub_c.exp().sum().backward();
                assert_eq!(sub_g.get(&sub_a).as_array(), g_a[i][j]);
                assert_eq!(sub_g.get(&sub_b).as_array(), g_b[i][j]);
            }
        }
    }

    #[test]
    fn test_matmul_vec_normal() {
        let dev = build_test_device!();

        let a = dev.tensor([0.7296, 0.3974, 0.9487]);
        let b = dev.tensor([[0.7804, 0.5540], [0.5378, 0.8401], [0.5042, 0.8604]]);
        let r = a.trace().matmul(b.clone());
        assert_close(&r.as_array(), &[1.261436, 1.5543157]);
        let g = r.exp().mean().backward();
        assert_close(&g.get(&a).as_array(), &[2.6883178, 2.9369607, 2.9256766]);
        assert_close(
            &g.get(&b).as_array(),
            &[
                [1.2879219, 1.7261779],
                [0.70150787, 0.94021803],
                [1.6746868, 2.244552],
            ],
        );
    }

    #[test]
    fn test_matmul_vec_transpose() {
        let dev = build_test_device!();
        let a = dev.tensor([0.7296, 0.3974, 0.9487]);
        let b = dev.tensor([[0.7804, 0.5378, 0.5042], [0.5540, 0.8401, 0.8604]]);
        let r = a.trace().matmul(b.trace().permute());
        assert_close(&r.as_array(), &[1.261436, 1.5543157]);
        let g = r.exp().mean().backward();
        assert_close(&g.get(&a).as_array(), &[2.6883178, 2.9369607, 2.9256766]);
        assert_close(
            &g.get(&b).as_array(),
            &[
                [1.2879219, 0.70150787, 1.6746868],
                [1.7261779, 0.94021803, 2.244552],
            ],
        );
    }

    #[test]
    fn test_vecvec() {
        let dev = build_test_device!();
        let a = dev.tensor([-1.5333828, 0.6136148, -0.77502704, -1.0014728, -2.0131118]);
        let b = dev.tensor([0.43068963, -0.9757187, -0.50650096]);
        let c = a.trace().matmul(b.clone());
        assert_close(
            &c.as_array(),
            &[
                [-0.66041213, 1.4961504, 0.7766599],
                [0.26427752, -0.5987154, -0.31079647],
                [-0.3337961, 0.75620836, 0.39255193],
                [-0.43132398, 0.97715575, 0.507247],
                [-0.8670264, 1.9642308, 1.0196431],
            ],
        );

        let g = c.exp().mean().backward();
        assert_close(
            &g.get(&a).as_array(),
            &[
                -0.34898597,
                -0.02309341,
                -0.16800028,
                -0.21024881,
                -0.54529756,
            ],
        );
        assert_close(
            &g.get(&b).as_array(),
            &[-0.13630435, -1.6781758, -0.75171506],
        );
    }
}
