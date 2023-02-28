#![allow(clippy::type_complexity)]

pub(super) mod cpu_kernel;

#[cfg(feature = "cuda")]
pub(super) mod cuda_kernel;

use crate::{
    gradients::{Merge, Tape},
    shapes::{Const, Dim, Dtype, Shape},
    tensor::{DeviceStorage, HasErr, PutTape, SplitTape, Tensor},
};

/// Matrix * Matrix, Vector * Matrix, Vector * Vector, and broadcasted/batched versions.
///
/// # Examples
/// 1. Matrix & Matrix
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<Rank2<3, 10>, f32, _> = dev.zeros();
/// let y: Tensor<Rank2<10, 5>, f32, _> = dev.zeros();
/// let _: Tensor<Rank2<3, 5>, f32, _> = x.matmul(y);
/// ```
///
/// 2. Vector x Matrix
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<Rank1<2>, f32, _> = dev.zeros();
/// let y: Tensor<Rank2<2, 4>, f32, _> = dev.zeros();
/// let _: Tensor<Rank1<4>, f32, _> = x.matmul(y);
/// ```
///
/// 3. Vector x Vector
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<Rank1<2>, f32, _> = dev.zeros();
/// let y: Tensor<Rank1<4>, f32, _> = dev.zeros();
/// let _: Tensor<Rank2<2, 4>, f32, _> = x.matmul(y);
/// ```
///
/// 4. Batched matmul
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<Rank3<10, 3, 2>, f32, _> = dev.zeros();
/// let y: Tensor<Rank3<10, 2, 4>, f32, _> = dev.zeros();
/// let _: Tensor<Rank3<10, 3, 4>, f32, _> = x.matmul(y);
/// ```
///
/// 5. Broadcasted matmul
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<Rank3<10, 3, 2>, f32, _> = dev.zeros();
/// let y: Tensor<Rank2<2, 4>, f32, _> = dev.zeros();
/// let _: Tensor<Rank3<10, 3, 4>, f32, _> = x.matmul(y);
/// ```
pub fn matmul<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where
    Lhs: TryMatMul<Rhs>,
{
    lhs.matmul(rhs)
}

/// Fallible matrix multiplication. See [matmul] for examples.
pub trait TryMatMul<Rhs>: HasErr {
    type Output;
    fn matmul(self, rhs: Rhs) -> Self::Output {
        self.try_matmul(rhs).unwrap()
    }
    fn try_matmul(self, rhs: Rhs) -> Result<Self::Output, Self::Err>;
}

#[rustfmt::skip]
fn try_binary_op<
    Lhs: Shape,
    Rhs: Shape,
    Out: Shape,
    E: Dtype,
    D: DeviceStorage,
    RhsTape: Tape<D>,
    LhsTape: Tape<D> + Merge<RhsTape>,
    Fwd: 'static + FnMut(&D, &D::Storage<Lhs, E>, &D::Storage<Rhs, E>) -> Result<D::Storage<Out, E>, D::Err>,
    Bwd: 'static + FnMut(&D, &D::Storage<Lhs, E>, &mut D::Storage<Lhs, E>, &D::Storage<Rhs, E>, &mut D::Storage<Rhs, E>, &D::Storage<Out, E>) -> Result<(), D::Err>,
>(
    lhs: Tensor<Lhs, E, D, LhsTape>,
    rhs: Tensor<Rhs, E, D, RhsTape>,
    mut fwd: Fwd,
    mut bwd: Bwd,
) -> Result<Tensor<Out, E, D, LhsTape>, D::Err> {
    let (lhs, ltape) = lhs.split_tape();
    let (rhs, rtape) = rhs.split_tape();
    let mut tape = ltape.merge(rtape);
    let out = lhs.device.upgrade(fwd(&lhs.device, &lhs.storage, &rhs.storage)?);
    let phantom_out = out.clone();
    tape.try_alloc_grad(&lhs)?;
    tape.try_alloc_grad(&rhs)?;
    tape.try_alloc_grad(&out)?;
    tape.add_backward_op(move |grads| {
        let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out);
        bwd(&lhs.device, &lhs.storage, grad_lhs, &rhs.storage, grad_rhs, grad_out)
    });
    Ok(out.put_tape(tape))
}

pub trait VecVecKernel<E: Dtype>: DeviceStorage {
    fn forward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), E>,
        rhs: &Self::Storage<(N,), E>,
    ) -> Result<Self::Storage<(M, N), E>, Self::Err>;

    fn backward<M: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(M,), E>,
        grad_lhs: &mut Self::Storage<(M,), E>,
        rhs: &Self::Storage<(N,), E>,
        grad_rhs: &mut Self::Storage<(N,), E>,
        grad_out: &Self::Storage<(M, N), E>,
    ) -> Result<(), Self::Err>;
}

impl<M: Dim, N: Dim, E: Dtype, D: VecVecKernel<E>, T: Tape<D> + Merge<R>, R: Tape<D>>
    TryMatMul<Tensor<(N,), E, D, R>> for Tensor<(M,), E, D, T>
{
    type Output = Tensor<(M, N), E, D, T>;
    fn try_matmul(self, rhs: Tensor<(N,), E, D, R>) -> Result<Self::Output, Self::Err> {
        try_binary_op(self, rhs, D::forward, D::backward)
    }
}

pub trait VecMatKernel<E: Dtype>: DeviceStorage {
    fn forward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), E>,
        rhs: &Self::Storage<(Const<K>, N), E>,
    ) -> Result<Self::Storage<(N,), E>, Self::Err>;

    fn backward<const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<K>,), E>,
        grad_lhs: &mut Self::Storage<(Const<K>,), E>,
        rhs: &Self::Storage<(Const<K>, N), E>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), E>,
        grad_out: &Self::Storage<(N,), E>,
    ) -> Result<(), Self::Err>;
}

impl<const K: usize, N: Dim, E: Dtype, D: VecMatKernel<E>, T: Tape<D> + Merge<R>, R: Tape<D>>
    TryMatMul<Tensor<(Const<K>, N), E, D, R>> for Tensor<(Const<K>,), E, D, T>
{
    type Output = Tensor<(N,), E, D, T>;
    fn try_matmul(self, rhs: Tensor<(Const<K>, N), E, D, R>) -> Result<Self::Output, Self::Err> {
        try_binary_op(self, rhs, D::forward, D::backward)
    }
}

pub trait MatMatKernel<E: Dtype>: DeviceStorage {
    fn forward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), E>,
        rhs: &Self::Storage<(Const<K>, N), E>,
    ) -> Result<Self::Storage<(M, N), E>, Self::Err>;

    fn backward<M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(M, Const<K>), E>,
        grad_lhs: &mut Self::Storage<(M, Const<K>), E>,
        rhs: &Self::Storage<(Const<K>, N), E>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), E>,
        grad_out: &Self::Storage<(M, N), E>,
    ) -> Result<(), Self::Err>;
}

impl<M: Dim, const K: usize, N: Dim, E: Dtype, D: MatMatKernel<E>, T, R>
    TryMatMul<Tensor<(Const<K>, N), E, D, R>> for Tensor<(M, Const<K>), E, D, T>
where
    T: Tape<D> + Merge<R>,
    R: Tape<D>,
{
    type Output = Tensor<(M, N), E, D, T>;
    fn try_matmul(self, rhs: Tensor<(Const<K>, N), E, D, R>) -> Result<Self::Output, Self::Err> {
        try_binary_op(self, rhs, D::forward, D::backward)
    }
}

pub trait MatMatBrKernel<E: Dtype>: DeviceStorage {
    fn forward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), E>,
        rhs: &Self::Storage<(Const<K>, N), E>,
    ) -> Result<Self::Storage<(B, M, N), E>, Self::Err>;

    fn backward<B: Dim, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(B, M, Const<K>), E>,
        grad_lhs: &mut Self::Storage<(B, M, Const<K>), E>,
        rhs: &Self::Storage<(Const<K>, N), E>,
        grad_rhs: &mut Self::Storage<(Const<K>, N), E>,
        grad_out: &Self::Storage<(B, M, N), E>,
    ) -> Result<(), Self::Err>;
}

impl<B: Dim, M: Dim, const K: usize, N: Dim, E: Dtype, D: MatMatBrKernel<E>, T, R>
    TryMatMul<Tensor<(Const<K>, N), E, D, R>> for Tensor<(B, M, Const<K>), E, D, T>
where
    T: Tape<D> + Merge<R>,
    R: Tape<D>,
{
    type Output = Tensor<(B, M, N), E, D, T>;
    fn try_matmul(self, rhs: Tensor<(Const<K>, N), E, D, R>) -> Result<Self::Output, Self::Err> {
        try_binary_op(self, rhs, D::forward, D::backward)
    }
}

pub trait MatMatBatch3Kernel<E: Dtype>: DeviceStorage {
    fn forward<const B: usize, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, K), E>,
        rhs: &Self::Storage<(Const<B>, K, N), E>,
    ) -> Result<Self::Storage<(Const<B>, M, N), E>, Self::Err>;

    fn backward<const B: usize, M: Dim, K: Dim, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, M, K), E>,
        grad_lhs: &mut Self::Storage<(Const<B>, M, K), E>,
        rhs: &Self::Storage<(Const<B>, K, N), E>,
        grad_rhs: &mut Self::Storage<(Const<B>, K, N), E>,
        grad_out: &Self::Storage<(Const<B>, M, N), E>,
    ) -> Result<(), Self::Err>;
}

impl<const B: usize, M: Dim, K: Dim, N: Dim, E: Dtype, D, T, R>
    TryMatMul<Tensor<(Const<B>, K, N), E, D, R>> for Tensor<(Const<B>, M, K), E, D, T>
where
    D: MatMatBatch3Kernel<E>,
    T: Tape<D> + Merge<R>,
    R: Tape<D>,
{
    type Output = Tensor<(Const<B>, M, N), E, D, T>;
    fn try_matmul(self, rhs: Tensor<(Const<B>, K, N), E, D, R>) -> Result<Self::Output, Self::Err> {
        try_binary_op(self, rhs, D::forward, D::backward)
    }
}

pub trait MatMatBatch4Kernel<E: Dtype>: DeviceStorage {
    fn forward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), E>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), E>,
    ) -> Result<Self::Storage<(Const<B>, Const<S>, M, N), E>, Self::Err>;

    fn backward<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim>(
        &self,
        lhs: &Self::Storage<(Const<B>, Const<S>, M, Const<K>), E>,
        grad_lhs: &mut Self::Storage<(Const<B>, Const<S>, M, Const<K>), E>,
        rhs: &Self::Storage<(Const<B>, Const<S>, Const<K>, N), E>,
        grad_rhs: &mut Self::Storage<(Const<B>, Const<S>, Const<K>, N), E>,
        grad_out: &Self::Storage<(Const<B>, Const<S>, M, N), E>,
    ) -> Result<(), Self::Err>;
}

impl<const B: usize, const S: usize, M: Dim, const K: usize, N: Dim, E: Dtype, D, T, R>
    TryMatMul<Tensor<(Const<B>, Const<S>, Const<K>, N), E, D, R>>
    for Tensor<(Const<B>, Const<S>, M, Const<K>), E, D, T>
where
    D: MatMatBatch4Kernel<E>,
    T: Tape<D> + Merge<R>,
    R: Tape<D>,
{
    type Output = Tensor<(Const<B>, Const<S>, M, N), E, D, T>;
    fn try_matmul(
        self,
        rhs: Tensor<(Const<B>, Const<S>, Const<K>, N), E, D, R>,
    ) -> Result<Self::Output, Self::Err> {
        try_binary_op(self, rhs, D::forward, D::backward)
    }
}

/// Utility function returning the ld and whether the matrix is transposed
/// for cublas & cblas.
#[allow(unused)]
pub(super) fn matrix_strides((m, n): (usize, usize), strides: [usize; 2]) -> (usize, bool) {
    match strides {
        [1, 0] => (m, true),
        [0, 1] => (n, false),
        [1, 1] => (n, false),
        [ld, 1] => (ld, false),
        [1, ld] => (ld, true),
        _ => panic!("At least a single stride must be 1 for cublas"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shapes::*, tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_valid_matmuls() {
        let dev: TestDevice = Default::default();

        {
            let a: Tensor<Rank1<3>, TestDtype, _> = dev.zeros();
            let b: Tensor<Rank2<3, 2>, TestDtype, _> = dev.zeros();
            let _: Tensor<Rank1<2>, TestDtype, _> = a.matmul(b);
        }

        {
            let a: Tensor<Rank2<5, 3>, TestDtype, _> = dev.zeros();
            let b: Tensor<Rank2<3, 2>, TestDtype, _> = dev.zeros();
            let _: Tensor<Rank2<5, 2>, TestDtype, _> = a.matmul(b);
        }

        {
            let a: Tensor<Rank3<10, 5, 3>, TestDtype, _> = dev.zeros();
            let b: Tensor<Rank2<3, 2>, TestDtype, _> = dev.zeros();
            let _: Tensor<Rank3<10, 5, 2>, TestDtype, _> = a.matmul(b);
        }

        {
            let a: Tensor<Rank3<10, 5, 3>, TestDtype, _> = dev.zeros();
            let b: Tensor<Rank3<10, 3, 2>, TestDtype, _> = dev.zeros();
            let _: Tensor<Rank3<10, 5, 2>, TestDtype, _> = a.matmul(b);
        }

        {
            let a: Tensor<Rank4<10, 20, 5, 3>, TestDtype, _> = dev.zeros();
            let b: Tensor<Rank4<10, 20, 3, 2>, TestDtype, _> = dev.zeros();
            let _: Tensor<Rank4<10, 20, 5, 2>, TestDtype, _> = a.matmul(b);
        }
    }

    #[test]
    fn test_matmul_normal() {
        let dev: TestDevice = Default::default();

        let a: Tensor<_, TestDtype, _> = dev.tensor([
            [0.5086, 0.5234, 0.2684],
            [0.8075, 0.8437, 0.9951],
            [0.0774, 0.7539, 0.8894],
            [0.8119, 0.2693, 0.7249],
        ]);
        let b: Tensor<_, TestDtype, _> =
            dev.tensor([[0.4651, 0.9106], [0.3360, 0.5534], [0.8092, 0.3827]]);
        let r = a.trace().matmul(b.clone());
        assert_close(
            &r.array(),
            &[
                [0.62960154, 0.8554974],
                [1.4642863, 1.5830379],
                [1.0090116, 0.82806206],
                [1.0546886, 1.165766],
            ],
        );
        let g = r.exp().mean().backward();
        assert_close(
            &g.get(&a).array(),
            &[
                [0.37689444, 0.24156547, 0.30238447],
                [0.80570966, 0.5184905, 0.6703743],
                [0.4199963, 0.2735345, 0.38693744],
                [0.5321113, 0.34252504, 0.4438907],
            ],
        );
        assert_close(
            &g.get(&b).array(),
            &[
                [0.8737376, 0.9888564],
                [0.9339924, 0.991189],
                [1.1659734, 1.2298465],
            ],
        );
    }

    #[test]
    fn test_matmul_transpose() {
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank2<4, 3>, TestDtype, _> = dev.sample_normal();
        let b: Tensor<Rank2<3, 2>, TestDtype, _> = dev.sample_normal();

        let c = a.trace().matmul(b.clone());
        let g1 = c.exp().mean().backward();

        let c2 = b.trace().permute().matmul(a.trace().permute());
        let g2 = c2.exp().mean().backward();

        assert_close(&g1.get(&a).array(), &g2.get(&a).array());
        assert_close(&g1.get(&b).array(), &g2.get(&b).array());
    }

    #[test]
    fn test_matmul_broadcast() {
        const N: usize = 5;
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank3<N, 4, 3>, TestDtype, _> = dev.sample_normal();
        let a_array = a.array();
        let b: Tensor<Rank2<3, 2>, TestDtype, _> = dev.sample_normal();
        let r = a.trace().matmul(b.clone());
        let r_array = r.array();
        for i in 0..N {
            let sub_a = dev.tensor(a_array[i]);
            let sub_c = sub_a.matmul(b.clone());
            assert_close(&r_array[i], &sub_c.array());
        }
        let gs = r.sum().backward();
        let a_grad = gs.get(&a).array();
        let mut sub_bs_summed = [[0.0; 2]; 3];
        for i in 0..N {
            let sub_a = dev.tensor(a_array[i]);
            let sub_gs = sub_a.trace().matmul(b.clone()).sum().backward();
            assert_close(&a_grad[i], &sub_gs.get(&sub_a).array());
            let sub_b_grad = sub_gs.get(&b).array();
            for x in 0..3 {
                for y in 0..2 {
                    sub_bs_summed[x][y] += sub_b_grad[x][y];
                }
            }
        }
        assert_close(&gs.get(&b).array(), &sub_bs_summed);
    }

    #[test]
    fn test_matmul_broadcast_actual() {
        const N: usize = 5;
        let dev: TestDevice = Default::default();
        let a: Tensor<Rank3<N, 4, 3>, TestDtype, _> = dev.sample_normal();
        let b: Tensor<Rank2<3, 2>, TestDtype, _> = dev.sample_normal();
        let b_up = dev.tensor([b.array(); N]);
        let r1 = a.trace().matmul(b_up.clone());
        let r2 = a.trace().matmul(b.clone());
        assert_eq!(r1.array(), r2.array());
        let g1 = r1.exp().mean().backward();
        let g2 = r2.exp().mean().backward();
        assert_eq!(g1.get(&a).array(), g2.get(&a).array());
        assert_close(
            &dev.tensor(g1.get(&b_up).array())
                .sum::<_, Axis<0>>()
                .array(),
            &g2.get(&b).array(),
        );
    }

    #[test]
    fn test_matmul_batched_3d() {
        let dev: TestDevice = Default::default();

        let a: Tensor<Rank3<5, 3, 2>, TestDtype, _> = dev.sample_normal();
        let a_array = a.array();
        let b: Tensor<Rank3<5, 2, 4>, TestDtype, _> = dev.sample_normal();
        let b_array = b.array();
        let c = a.trace().matmul(b.clone());
        let c_array = c.array();
        let g = c.exp().sum().backward();

        let g_a = g.get(&a).array();
        let g_b = g.get(&b).array();

        for i in 0..5 {
            let sub_a = dev.tensor(a_array[i]);
            let sub_b = dev.tensor(b_array[i]);
            let sub_c = sub_a.trace().matmul(sub_b.clone());
            assert_close(&sub_c.array(), &c_array[i]);
            let sub_g = sub_c.exp().sum().backward();
            assert_close(&sub_g.get(&sub_a).array(), &g_a[i]);
            sub_g.get(&sub_b).array().assert_close(&g_b[i], 1e-5);
        }
    }

    #[test]
    fn test_matmul_batched_4d() {
        let dev: TestDevice = Default::default();

        let a: Tensor<Rank4<7, 5, 3, 2>, TestDtype, _> = dev.sample_normal();
        let a_array = a.array();
        let b: Tensor<Rank4<7, 5, 2, 4>, TestDtype, _> = dev.sample_normal();
        let b_array = b.array();
        let c = a.trace().matmul(b.clone());
        let c_array = c.array();
        let g = c.exp().sum().backward();

        let g_a = g.get(&a).array();
        let g_b = g.get(&b).array();

        for i in 0..7 {
            for j in 0..5 {
                let sub_a = dev.tensor(a_array[i][j]);
                let sub_b = dev.tensor(b_array[i][j]);
                let sub_c = sub_a.trace().matmul(sub_b.clone());
                assert_close(&sub_c.array(), &c_array[i][j]);
                let sub_g = sub_c.exp().sum().backward();
                assert_close(&sub_g.get(&sub_a).array(), &g_a[i][j]);
                sub_g.get(&sub_b).array().assert_close(&g_b[i][j], 1e-5);
            }
        }
    }

    #[test]
    fn test_matmul_vec_normal() {
        let dev: TestDevice = Default::default();

        let a: Tensor<_, TestDtype, _> = dev.tensor([0.7296, 0.3974, 0.9487]);
        let b: Tensor<_, TestDtype, _> =
            dev.tensor([[0.7804, 0.5540], [0.5378, 0.8401], [0.5042, 0.8604]]);
        let r = a.trace().matmul(b.clone());
        assert_close(&r.array(), &[1.261436, 1.5543157]);
        let g = r.exp().mean().backward();
        assert_close(&g.get(&a).array(), &[2.6883178, 2.9369607, 2.9256766]);
        assert_close(
            &g.get(&b).array(),
            &[
                [1.2879219, 1.7261779],
                [0.70150787, 0.94021803],
                [1.6746868, 2.244552],
            ],
        );
    }

    #[test]
    fn test_matmul_vec_transpose() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([0.7296, 0.3974, 0.9487]);
        let b: Tensor<_, TestDtype, _> =
            dev.tensor([[0.7804, 0.5378, 0.5042], [0.5540, 0.8401, 0.8604]]);
        let r = a.trace().matmul(b.trace().permute());
        assert_close(&r.array(), &[1.261436, 1.5543157]);
        let g = r.exp().mean().backward();
        assert_close(&g.get(&a).array(), &[2.6883178, 2.9369607, 2.9256766]);
        assert_close(
            &g.get(&b).array(),
            &[
                [1.2879219, 0.70150787, 1.6746868],
                [1.7261779, 0.94021803, 2.244552],
            ],
        );
    }

    #[test]
    fn test_vecvec() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> =
            dev.tensor([-1.5333828, 0.6136148, -0.77502704, -1.0014728, -2.0131118]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([0.43068963, -0.9757187, -0.50650096]);
        let c = a.trace().matmul(b.clone());
        let c_t = b.trace().matmul(a.clone()).permute();
        assert_eq!(c.array(), c_t.array());
        assert_close(
            &c.array(),
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
            &g.get(&a).array(),
            &[
                -0.34898597,
                -0.02309341,
                -0.16800028,
                -0.21024881,
                -0.54529756,
            ],
        );
        assert_close(&g.get(&b).array(), &[-0.13630435, -1.6781758, -0.75171506]);
    }

    #[test]
    fn test_small_matmul_vv() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([0.5]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([2.0]);
        let c = a.trace().matmul(b.clone());
        assert_eq!(c.array(), [[1.0]]);
        let g = c.exp().sum().backward();
        assert_close(&g.get(&a).array(), &[5.4365635]);
        assert_close(&g.get(&b).array(), &[1.3591409]);
    }

    #[test]
    fn test_small_matmul_vm() {
        let dev: TestDevice = Default::default();

        // 1 * 1x1
        let a: Tensor<_, TestDtype, _> = dev.tensor([0.5]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([[2.0]]);
        let c = a.trace().matmul(b.clone());
        assert_eq!(c.array(), [1.0]);
        let g = c.exp().sum().backward();
        assert_close(&g.get(&a).array(), &[5.4365635]);
        assert_close(&g.get(&b).array(), &[[1.3591409]]);

        // 1 * 1x1 (permuted)
        let c = a.trace().matmul(b.trace().permute());
        assert_eq!(c.array(), [1.0]);
        let g = c.exp().sum().backward();
        assert_close(&g.get(&a).array(), &[5.4365635]);
        assert_close(&g.get(&b).array(), &[[1.3591409]]);

        // 1 * 1x2
        let a: Tensor<_, TestDtype, _> = dev.tensor([0.5]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([[2.0, 4.0]]);
        let c = a.trace().matmul(b.clone());
        let e = [1.0, 2.0];
        assert_eq!(c.array(), e);
        let g = c.exp().sum().backward();
        assert_close(&g.get(&a).array(), &[e[0].exp() * 2.0 + e[1].exp() * 4.0]);
        assert_close(&g.get(&b).array(), &[[1.3591409, 3.694528]]);

        // 1 * 1x2 (permuted)
        let a: Tensor<_, TestDtype, _> = dev.tensor([0.5]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([[2.0], [4.0]]);
        let c = a.trace().matmul(b.trace().permute());
        assert_eq!(c.array(), e);
        let g = c.exp().sum().backward();
        assert_close(&g.get(&a).array(), &[e[0].exp() * 2.0 + e[1].exp() * 4.0]);
        assert_close(&g.get(&b).array(), &[[1.3591409], [3.694528]]);
    }

    #[test]
    fn test_small_matmul_mm() {
        let dev: TestDevice = Default::default();

        {
            // 1x1 * 1x1
            let a: Tensor<_, TestDtype, _> = dev.tensor([[0.5]]);
            let b: Tensor<_, TestDtype, _> = dev.tensor([[2.0]]);
            let c = a.trace().matmul(b.clone());
            assert_eq!(c.array(), [[1.0]]);
            let g = c.exp().sum().backward();
            assert_close(&g.get(&a).array(), &[[5.4365635]]);
            assert_close(&g.get(&b).array(), &[[1.3591409]]);
        }

        {
            // 1x2 * 2x1
            let a: Tensor<_, TestDtype, _> = dev.tensor([[0.5, 0.1]]);
            let b: Tensor<_, TestDtype, _> = dev.tensor([[2.0], [4.0]]);
            let c = a.trace().matmul(b.clone());
            assert_eq!(c.array(), [[1.4]]);
            let g = c.exp().sum().backward();
            g.get(&a).array().assert_close(&[[8.1104, 16.2208]], 1e-5);
            g.get(&b)
                .array()
                .assert_close(&[[2.0276], [0.40552002]], 1e-5);
        }

        {
            // 1x2 (permuted) * 2x1
            let a: Tensor<_, TestDtype, _> = dev.tensor([[0.5], [0.1]]);
            let b: Tensor<_, TestDtype, _> = dev.tensor([[2.0], [4.0]]);
            let c = a.trace().permute().matmul(b.clone());
            assert_eq!(c.array(), [[1.4]]);
            let g = c.exp().sum().backward();
            g.get(&a).array().assert_close(&[[8.1104], [16.2208]], 1e-5);
            g.get(&b)
                .array()
                .assert_close(&[[2.0276], [0.40552002]], 1e-5);
        }

        {
            // 1x2 * 2x1 (permuted)
            let a: Tensor<_, TestDtype, _> = dev.tensor([[0.5, 0.1]]);
            let b: Tensor<_, TestDtype, _> = dev.tensor([[2.0, 4.0]]);
            let c = a.trace().matmul(b.trace().permute());
            assert_eq!(c.array(), [[1.4]]);
            let g = c.exp().sum().backward();
            g.get(&a).array().assert_close(&[[8.1104, 16.2208]], 1e-5);
            g.get(&b)
                .array()
                .assert_close(&[[2.0276, 0.40552002]], 1e-5);
        }
    }
}
