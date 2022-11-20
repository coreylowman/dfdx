use super::device::{Cpu, StridedArray};
use crate::arrays::*;
use crate::devices::{binary_ops, device::*};

#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, cblas_sgemv as sgemv, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};

#[derive(Copy, Clone)]
struct View<S: Shape, E: Dtype> {
    ptr: *const E,
    strides: S::Concrete,
}

#[derive(Copy, Clone)]
struct ViewMut<S: Shape, E: Dtype> {
    ptr: *mut E,
    strides: S::Concrete,
}

impl<S: Shape, E: Dtype> StridedArray<S, E> {
    fn view(&self) -> View<S, E> {
        View {
            ptr: self.data.as_ptr(),
            strides: self.strides.0,
        }
    }

    fn view_mut(&mut self) -> ViewMut<S, E> {
        ViewMut {
            ptr: std::sync::Arc::make_mut(&mut self.data).as_mut_ptr(),
            strides: self.strides.0,
        }
    }
}

impl<D1: Dim, D2: Dim, E: Dtype> View<(D1, D2), E> {
    fn transpose(self) -> View<(D2, D1), E> {
        View {
            ptr: self.ptr,
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

impl<D1: Dim, D2: Dim, E: Dtype> ViewMut<(D1, D2), E> {
    fn transpose(self) -> ViewMut<(D2, D1), E> {
        ViewMut {
            ptr: self.ptr,
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, E: Dtype> View<(D1, D2, D3), E> {
    fn index(&self, index: usize) -> View<(D2, D3), E> {
        let [a, b, c] = self.strides;
        View {
            ptr: unsafe { self.ptr.add(index * a) },
            strides: [b, c],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, E: Dtype> ViewMut<(D1, D2, D3), E> {
    fn index(&self, index: usize) -> ViewMut<(D2, D3), E> {
        let [a, b, c] = self.strides;
        ViewMut {
            ptr: unsafe { self.ptr.add(index * a) },
            strides: [b, c],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, E: Dtype> View<(D1, D2, D3, D4), E> {
    fn index(&self, index: usize) -> View<(D2, D3, D4), E> {
        let [a, b, c, d] = self.strides;
        View {
            ptr: unsafe { self.ptr.add(index * a) },
            strides: [b, c, d],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, E: Dtype> ViewMut<(D1, D2, D3, D4), E> {
    fn index(&self, index: usize) -> ViewMut<(D2, D3, D4), E> {
        let [a, b, c, d] = self.strides;
        ViewMut {
            ptr: unsafe { self.ptr.add(index * a) },
            strides: [b, c, d],
        }
    }
}

fn matmul<const M: usize, const K: usize, const N: usize>(
    a: View<Rank2<M, K>, f32>,
    b: View<Rank2<K, N>, f32>,
    c: ViewMut<Rank2<M, N>, f32>,
) {
    #[cfg(not(feature = "cblas"))]
    unsafe {
        let [ar, ac] = a.strides.map(|x| x as isize);
        let [br, bc] = b.strides.map(|x| x as isize);
        let [cr, cc] = c.strides.map(|x| x as isize);
        matrixmultiply::sgemm(
            M, K, N, 1.0, a.ptr, ar, ac, b.ptr, br, bc, 1.0, c.ptr, cr, cc,
        );
    }

    #[cfg(feature = "cblas")]
    unsafe {
        let (m, n, k) = (M as libc::c_int, N as libc::c_int, K as libc::c_int);
        let [ar, ac] = a.strides.map(|x| x as libc::c_int);
        let [br, bc] = b.strides.map(|x| x as libc::c_int);
        let [cr, cc] = c.strides.map(|x| x as libc::c_int);
        let (layout, a_tr, b_tr, lda, ldb, ldc) = if cr < cc {
            let layout = ColMajor;
            let ldc = m;
            let (lda, a_tr) = {
                if ar < ac {
                    (m, NoTr)
                } else {
                    (k, Tr)
                }
            };
            let (ldb, b_tr) = {
                if br < bc {
                    (k, NoTr)
                } else {
                    (n, Tr)
                }
            };
            (layout, a_tr, b_tr, lda, ldb, ldc)
        } else {
            let layout = RowMajor;
            let ldc = n;
            let (lda, a_tr) = {
                if ar < ac {
                    (m, Tr)
                } else {
                    (k, NoTr)
                }
            };
            let (ldb, b_tr) = {
                if br < bc {
                    (k, Tr)
                } else {
                    (n, NoTr)
                }
            };
            (layout, a_tr, b_tr, lda, ldb, ldc)
        };
        sgemm(
            layout, a_tr, b_tr, m, n, k, 1.0, a.ptr, lda, b.ptr, ldb, 1.0, c.ptr, ldc,
        )
    }
}

fn matmul_bwd<const M: usize, const K: usize, const N: usize>(
    lhs: View<Rank2<M, K>, f32>,
    grad_lhs: ViewMut<Rank2<M, K>, f32>,
    rhs: View<Rank2<K, N>, f32>,
    grad_rhs: ViewMut<Rank2<K, N>, f32>,
    grad_out: View<Rank2<M, N>, f32>,
) {
    // grad_lhs += grad_out * rhs^T
    // (M, K) += (M, N) * (N, K)
    matmul(grad_out, rhs.transpose(), grad_lhs);

    // grad_rhs += lhs^T * grad_out
    // (K, N) += (K, M) * (M, N)
    matmul(lhs.transpose(), grad_out, grad_rhs);
}

impl<const M: usize, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank2<M, K>, Rank2<K, N>, Rank2<M, N>, f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank2<M, K>, f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
    ) -> Result<Self::Storage<Rank2<M, N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank2<M, N>, f32> = self.try_zeros()?;
        matmul(lhs.view(), rhs.view(), out.view_mut());
        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank2<M, K>, f32>,
        grad_lhs: &mut Self::Storage<Rank2<M, K>, f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
        grad_rhs: &mut Self::Storage<Rank2<K, N>, f32>,
        grad_out: &Self::Storage<Rank2<M, N>, f32>,
    ) {
        matmul_bwd(
            lhs.view(),
            grad_lhs.view_mut(),
            rhs.view(),
            grad_rhs.view_mut(),
            grad_out.view(),
        );
    }
}

impl<const B: usize, const M: usize, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank3<B, M, K>, Rank3<B, K, N>, Rank3<B, M, N>, f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank3<B, M, K>, f32>,
        rhs: &Self::Storage<Rank3<B, K, N>, f32>,
    ) -> Result<Self::Storage<Rank3<B, M, N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank3<B, M, N>, f32> = self.try_zeros()?;

        let a = lhs.view();
        let b = rhs.view();
        let c = out.view_mut();

        for batch in 0..B {
            matmul(a.index(batch), b.index(batch), c.index(batch));
        }

        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank3<B, M, K>, f32>,
        grad_lhs: &mut Self::Storage<Rank3<B, M, K>, f32>,
        rhs: &Self::Storage<Rank3<B, K, N>, f32>,
        grad_rhs: &mut Self::Storage<(C<B>, C<K>, C<N>), f32>,
        grad_out: &Self::Storage<Rank3<B, M, N>, f32>,
    ) {
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for batch in 0..B {
            matmul_bwd(
                lhs.index(batch),
                grad_lhs.index(batch),
                rhs.index(batch),
                grad_rhs.index(batch),
                grad_out.index(batch),
            );
        }
    }
}

impl<Batch: Dim, const M: usize, const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, (Batch, C<M>, C<K>), (C<K>, C<N>), (Batch, C<M>, C<N>), f32>
    for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(C<K>, C<N>), f32>,
    ) -> Result<Self::Storage<(Batch, C<M>, C<N>), f32>, Self::Err> {
        let batch_size = lhs.shape().0.size();

        let mut out: Self::Storage<(Batch, C<M>, C<N>), f32> =
            self.try_zeros_like((lhs.shape().0, C, C))?;

        let a = lhs.view();
        let b = rhs.view();
        let c = out.view_mut();

        for batch in 0..batch_size {
            matmul(a.index(batch), b, c.index(batch));
        }

        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(Batch, C<M>, C<K>), f32>,
        grad_lhs: &mut Self::Storage<(Batch, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(C<K>, C<N>), f32>,
        grad_rhs: &mut Self::Storage<(C<K>, C<N>), f32>,
        grad_out: &Self::Storage<(Batch, C<M>, C<N>), f32>,
    ) {
        let batch_size = lhs.shape().0.size();
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for batch in 0..batch_size {
            matmul_bwd(
                lhs.index(batch),
                grad_lhs.index(batch),
                rhs,
                grad_rhs,
                grad_out.index(batch),
            );
        }
    }
}

impl<const B: usize, const S: usize, const M: usize, const K: usize, const N: usize>
    BinaryKernel<
        binary_ops::MatMul,
        (C<B>, C<S>, C<M>, C<K>),
        (C<B>, C<S>, C<K>, C<N>),
        (C<B>, C<S>, C<M>, C<N>),
        f32,
    > for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(C<B>, C<S>, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(C<B>, C<S>, C<K>, C<N>), f32>,
    ) -> Result<Self::Storage<(C<B>, C<S>, C<M>, C<N>), f32>, Self::Err> {
        let mut out: Self::Storage<(C<B>, C<S>, C<M>, C<N>), f32> = self.try_zeros()?;
        let lhs = lhs.view();
        let rhs = rhs.view();
        let out_view = out.view_mut();
        for b in 0..B {
            for s in 0..S {
                matmul(
                    lhs.index(b).index(s),
                    rhs.index(b).index(s),
                    out_view.index(b).index(s),
                );
            }
        }
        Ok(out)
    }

    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<(C<B>, C<S>, C<M>, C<K>), f32>,
        grad_lhs: &mut Self::Storage<(C<B>, C<S>, C<M>, C<K>), f32>,
        rhs: &Self::Storage<(C<B>, C<S>, C<K>, C<N>), f32>,
        grad_rhs: &mut Self::Storage<(C<B>, C<S>, C<K>, C<N>), f32>,
        grad_out: &Self::Storage<(C<B>, C<S>, C<M>, C<N>), f32>,
    ) {
        let lhs = lhs.view();
        let grad_lhs = grad_lhs.view_mut();
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        for b in 0..B {
            for s in 0..S {
                matmul_bwd(
                    lhs.index(b).index(s),
                    grad_lhs.index(b).index(s),
                    rhs.index(b).index(s),
                    grad_rhs.index(b).index(s),
                    grad_out.index(b).index(s),
                );
            }
        }
    }
}

impl<const K: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank1<K>, Rank2<K, N>, Rank1<N>, f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank1<K>, f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
    ) -> Result<Self::Storage<Rank1<N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank1<N>, f32> = self.try_zeros()?;
        let lhs: View<(C<1>, C<K>), f32> = View {
            ptr: lhs.data.as_ptr(),
            strides: [0, lhs.strides.0[0]],
        };
        let rhs = rhs.view();
        let out_view = out.view_mut();
        let out_view: ViewMut<(C<1>, C<N>), f32> = ViewMut {
            ptr: out_view.ptr,
            strides: [0, out_view.strides[0]],
        };
        matmul(lhs, rhs, out_view);
        Ok(out)
    }
    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank1<K>, f32>,
        grad_lhs: &mut Self::Storage<Rank1<K>, f32>,
        rhs: &Self::Storage<Rank2<K, N>, f32>,
        grad_rhs: &mut Self::Storage<Rank2<K, N>, f32>,
        grad_out: &Self::Storage<Rank1<N>, f32>,
    ) {
        let lhs: View<(C<1>, C<K>), f32> = View {
            ptr: lhs.data.as_ptr(),
            strides: [0, lhs.strides.0[0]],
        };
        let grad_lhs = grad_lhs.view_mut();
        let grad_lhs: ViewMut<(C<1>, C<K>), f32> = ViewMut {
            ptr: grad_lhs.ptr,
            strides: [0, grad_lhs.strides[0]],
        };
        let rhs = rhs.view();
        let grad_rhs = grad_rhs.view_mut();
        let grad_out = grad_out.view();
        let grad_out: View<(C<1>, C<N>), f32> = View {
            ptr: grad_out.ptr,
            strides: [0, grad_out.strides[0]],
        };
        matmul_bwd(lhs, grad_lhs, rhs, grad_rhs, grad_out);
    }
}

impl<const M: usize, const N: usize>
    BinaryKernel<binary_ops::MatMul, Rank1<M>, Rank1<N>, Rank2<M, N>, f32> for Cpu
{
    fn binary_fwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank1<M>, f32>,
        rhs: &Self::Storage<Rank1<N>, f32>,
    ) -> Result<Self::Storage<Rank2<M, N>, f32>, Self::Err> {
        let mut out: Self::Storage<Rank2<M, N>, f32> = self.try_zeros()?;
        let lhs: View<(C<M>, C<1>), f32> = View {
            ptr: lhs.data.as_ptr(),
            strides: [lhs.strides.0[0], 0],
        };
        let rhs: View<(C<1>, C<N>), f32> = View {
            ptr: rhs.data.as_ptr(),
            strides: [0, rhs.strides.0[0]],
        };
        let out_view = out.view_mut();
        matmul(lhs, rhs, out_view);
        Ok(out)
    }
    fn binary_bwd(
        &self,
        _op: binary_ops::MatMul,
        lhs: &Self::Storage<Rank1<M>, f32>,
        grad_lhs: &mut Self::Storage<Rank1<M>, f32>,
        rhs: &Self::Storage<Rank1<N>, f32>,
        grad_rhs: &mut Self::Storage<Rank1<N>, f32>,
        grad_out: &Self::Storage<Rank2<M, N>, f32>,
    ) {
        let lhs = lhs.view();
        let lhs: View<(C<M>, C<1>), f32> = View {
            ptr: lhs.ptr,
            strides: [lhs.strides[0], 0],
        };
        let grad_lhs = grad_lhs.view_mut();
        let grad_lhs: ViewMut<(C<M>, C<1>), f32> = ViewMut {
            ptr: grad_lhs.ptr,
            strides: [grad_lhs.strides[0], 0],
        };
        let rhs = rhs.view();
        let rhs: View<(C<1>, C<N>), f32> = View {
            ptr: rhs.ptr,
            strides: [0, rhs.strides[0]],
        };
        let grad_rhs = grad_rhs.view_mut();
        let grad_rhs: ViewMut<(C<1>, C<N>), f32> = ViewMut {
            ptr: grad_rhs.ptr,
            strides: [0, grad_rhs.strides[0]],
        };
        let grad_out = grad_out.view();
        matmul_bwd(lhs, grad_lhs, rhs, grad_rhs, grad_out);
    }
}
