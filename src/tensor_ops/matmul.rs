use crate::{
    arrays::{Dtype, Shape},
    devices::{
        binary_ops,
        device::{BinaryKernel, HasErr},
        Device,
    },
    gradients::{Merge, Tape},
    tensor::Tensor,
};

use super::utils::try_binary_op;

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
pub trait TryMatMul<Rhs, Out>: HasErr {
    fn matmul(self, rhs: Rhs) -> Out {
        self.try_matmul(rhs).unwrap()
    }
    fn try_matmul(self, rhs: Rhs) -> Result<Out, Self::Err>;
}

impl<
        Lhs: Shape,
        Rhs: Shape,
        Out: Shape,
        E: Dtype,
        D: Device,
        LhsTape: Tape<D>,
        RhsTape: Tape<D>,
    > TryMatMul<Tensor<Rhs, E, D, RhsTape>, Tensor<Out, E, D, LhsTape>>
    for Tensor<Lhs, E, D, LhsTape>
where
    D: BinaryKernel<binary_ops::MatMul, Lhs, Rhs, Out, E>,
    LhsTape: Merge<RhsTape>,
{
    fn try_matmul(
        self,
        rhs: Tensor<Rhs, E, D, RhsTape>,
    ) -> Result<Tensor<Out, E, D, LhsTape>, Self::Err> {
        try_binary_op(Default::default(), self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{AsArray, Zeros};
    use crate::tensor::*;
    use crate::tensor_ops::impl_backward::TryBackward;
    use crate::tensor_ops::impl_mean::MeanTo;
    use crate::tensor_ops::map::TryExp;
    use crate::{
        devices::Device,
        tests::{assert_close, build_test_device},
    };

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
    fn test_matmul() {
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

    // #[test]
    // fn test_matmul_transpose() {
    //     let mut rng = thread_rng();
    //     let a: Tensor2D<4, 3> = TensorCreator::randn(&mut rng);
    //     let b: Tensor2D<3, 2> = TensorCreator::randn(&mut rng);
    //     let c = matmul(a.trace(), b.clone());

    //     let b_t = Tensor2D::new(transpose(b.data()));
    //     let c_tr = matmul_transpose(a.trace(), b_t.clone());
    //     assert_close(c_tr.data(), c.data());

    //     let gs = backward(c.exp().mean());
    //     let gs_tr = backward(c_tr.exp().mean());
    //     assert_close(gs_tr.ref_gradient(&a), gs.ref_gradient(&a));
    //     assert_close(gs_tr.ref_gradient(&b_t), &transpose(gs.ref_gradient(&b)));
    // }

    // #[test]
    // fn test_broadcasted_matmul() {
    //     const N: usize = 5;
    //     let mut rng = thread_rng();
    //     let a: Tensor3D<N, 4, 3> = TensorCreator::randn(&mut rng);
    //     let b: Tensor2D<3, 2> = TensorCreator::randn(&mut rng);
    //     let r = matmul(a.trace(), b.clone());
    //     for i in 0..N {
    //         let sub_a = Tensor2D::new(a.data()[i]);
    //         assert_close(&r.data()[i], matmul(sub_a, b.clone()).data());
    //     }
    //     let gs = backward(r.sum());
    //     let mut sub_bs_summed = [[0.0; 2]; 3];
    //     for i in 0..N {
    //         let sub_a = Tensor2D::new(a.data()[i]);
    //         let sub_gs = backward(matmul(sub_a.trace(), b.clone()).sum());
    //         assert_close(&gs.ref_gradient(&a)[i], sub_gs.ref_gradient(&sub_a));
    //         <Cpu as Device<_>>::add(&mut sub_bs_summed, sub_gs.ref_gradient(&b));
    //     }
    //     assert_close(gs.ref_gradient(&b), &sub_bs_summed);
    // }

    // #[test]
    // fn test_broadcasted_matmul_transpose() {
    //     let mut rng = thread_rng();
    //     let a: Tensor3D<2, 4, 3> = TensorCreator::randn(&mut rng);
    //     let b: Tensor2D<3, 2> = TensorCreator::randn(&mut rng);
    //     let c = matmul(a.trace(), b.clone());

    //     let b_t = Tensor2D::new(transpose(b.data()));
    //     let c_tr = matmul_transpose(a.trace(), b_t.clone());
    //     assert_close(c_tr.data(), c.data());

    //     let gs = backward(c.exp().mean());
    //     let gs_tr = backward(c_tr.exp().mean());
    //     assert_close(gs_tr.ref_gradient(&a), gs.ref_gradient(&a));
    //     assert_close(gs_tr.ref_gradient(&b_t), &transpose(gs.ref_gradient(&b)));
    // }

    // #[test]
    // fn test_vecmat_mul() {
    //     let a = tensor([0.7296, 0.3974, 0.9487]);
    //     let b = tensor([[0.7804, 0.5540], [0.5378, 0.8401], [0.5042, 0.8604]]);
    //     let r: Tensor1D<2, OwnedTape> = vecmat_mul(a.trace(), b.clone());
    //     assert_close(r.data(), &[1.261436, 1.5543157]);
    //     let g = backward(r.exp().mean());
    //     assert_close(g.ref_gradient(&a), &[2.6883178, 2.9369607, 2.9256766]);
    //     assert_close(
    //         g.ref_gradient(&b),
    //         &[
    //             [1.2879219, 1.7261779],
    //             [0.70150787, 0.94021803],
    //             [1.6746868, 2.244552],
    //         ],
    //     );
    // }

    // #[test]
    // fn test_vecmat_mul_transpose() {
    //     let a = tensor([0.7296, 0.3974, 0.9487]);
    //     let b = tensor([[0.7804, 0.5378, 0.5042], [0.5540, 0.8401, 0.8604]]);
    //     let r: Tensor1D<2, OwnedTape> = vecmat_mul_transpose(a.trace(), b.clone());
    //     assert_close(r.data(), &[1.261436, 1.5543157]);
    //     let g = backward(r.exp().mean());
    //     assert_close(g.ref_gradient(&a), &[2.6883178, 2.9369607, 2.9256766]);
    //     assert_close(
    //         g.ref_gradient(&b),
    //         &[
    //             [1.2879219, 0.70150787, 1.6746868],
    //             [1.7261779, 0.94021803, 2.244552],
    //         ],
    //     );
    // }

    // fn transpose<const M: usize, const N: usize>(a: &[[f32; N]; M]) -> [[f32; M]; N] {
    //     let mut t: [[f32; M]; N] = ZeroElements::ZEROS;
    //     for (m, a_m) in a.iter().enumerate() {
    //         for (n, a_mn) in a_m.iter().enumerate() {
    //             t[n][m] = *a_mn;
    //         }
    //     }
    //     t
    // }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::tests::assert_close;

//     #[test]
//     fn test_valid_matmuls() {
//         type A = [[f32; 2]; 5];
//         type B = [[f32; 3]; 2];
//         type C = [[f32; 3]; 5];

//         // normal matmul
//         let _ = <Cpu as MatMul<A, B, C>>::mm;

//         // batch 3d matmul
//         let _ = <Cpu as MatMul<[A; 10], [B; 10], [C; 10]>>::mm;

//         // batch 4d matmul
//         let _ = <Cpu as MatMul<[[A; 10]; 12], [[B; 10]; 12], [[C; 10]; 12]>>::mm;

//         // broadcast matmul
//         let _ = <Cpu as MatMul<[A; 10], B, [C; 10]>>::mm;
//         let _ = <Cpu as MatMul<[A; 10], [B; 10], C>>::mm;

//         // transposed
//         let _ = <Cpu as MatMul<C, <B as Transpose>::T, A>>::mm;
//         let _ = <Cpu as MatMul<<A as Transpose>::T, C, B>>::mm;
//     }

//     #[test]
//     fn test_matmul() {
//         let x = [
//             [1.0, 2.0, 3.0],
//             [4.0, 5.0, 6.0],
//             [7.0, 8.0, 9.0],
//             [10.0, 11.0, 12.0],
//         ];
//         let x_t = [
//             [1.0, 4.0, 7.0, 10.0],
//             [2.0, 5.0, 8.0, 11.0],
//             [3.0, 6.0, 9.0, 12.0],
//         ];
//         let y = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//         let y_t = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
//         let expected = [[22.0, 28.0], [49.0, 64.0], [76.0, 100.0], [103.0, 136.0]];

//         let mut out = [[0.0; 2]; 4];
//         Cpu::mm(&x, &y, &mut out);
//         assert_close(&out, &expected);

//         let mut out = [[0.0; 2]; 4];
//         Cpu::mm_at(&x_t, &y, &mut out);
//         assert_close(&out, &expected);

//         let mut out = [[0.0; 2]; 4];
//         Cpu::mm_bt(&x, &y_t, &mut out);
//         assert_close(&out, &expected);
//     }

//     #[test]
//     fn test_vecmul() {
//         let x = [1.0, 2.0, 3.0];
//         let y = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
//         let y_t = [[1.0, 0.5, 1.0 / 3.0], [2.0, 1.0, 1.0]];
//         let expected = [3.0, 7.0];

//         let mut out = [0.0; 2];
//         Cpu::vm(&x, &y, &mut out);
//         assert_close(&out, &expected);

//         let mut out = [0.0; 2];
//         Cpu::vm_bt(&x, &y_t, &mut out);
//         assert_close(&out, &expected);
//     }

//     #[test]
//     fn test_vecvec() {
//         let x = [1.0, 2.0, 3.0];
//         let y = [-1.0, 0.5, -1.0 / 3.0, 0.25];

//         let mut out = [[0.0; 4]; 3];
//         Cpu::vv(&x, &y, &mut out);
//         assert_close(
//             &out,
//             &[
//                 [-1.0, 0.5, -1.0 / 3.0, 0.25],
//                 [-2.0, 1.0, -2.0 / 3.0, 0.5],
//                 [-3.0, 1.5, -1.0, 0.75],
//             ],
//         );

//         let mut out = [[0.0; 3]; 4];
//         Cpu::vv(&y, &x, &mut out);
//         assert_close(
//             &out,
//             &[
//                 [-1.0, -2.0, -3.0],
//                 [0.5, 1.0, 1.5],
//                 [-1.0 / 3.0, -2.0 / 3.0, -1.0],
//                 [0.25, 0.5, 0.75],
//             ],
//         );
//     }
// }
