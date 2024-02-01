use crate::{
    shapes::*,
    tensor::{cpu::NdIndex, *},
};

impl<E: Dtype> super::ConcatAlongKernel<E> for Cpu {
    fn forward<A: Shape, B: Shape, C: Shape>(
        &self,
        ax: usize,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
        c: &mut Tensor<C, E, Self>,
    ) -> Result<(), Error> {
        let mut a_idx = NdIndex::new(a.shape, a.strides);
        let mut b_idx = NdIndex::new(b.shape, b.strides);

        let mut a_n = 1;
        let mut b_n = 1;
        for i in ax..A::NUM_DIMS {
            a_n *= a_idx.shape[i];
            b_n *= b_idx.shape[i];
        }

        let mut i = 0;
        let n = c.data.len();
        let buf = std::sync::Arc::get_mut(&mut c.data).unwrap();
        while i < n {
            for _ in 0..a_n {
                (*buf)[i] = a.data[a_idx.next().unwrap()];
                i += 1;
            }
            for _ in 0..b_n {
                (*buf)[i] = b.data[b_idx.next().unwrap()];
                i += 1;
            }
        }
        Ok(())
    }
    fn backward<A: Shape, B: Shape>(
        &self,
        ax: usize,
        a: &GhostTensor<A, E, Self>,
        grad_a: &mut Self::Vec,
        b: &GhostTensor<B, E, Self>,
        grad_b: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let mut a_idx = NdIndex::new(a.shape, a.strides);
        let mut b_idx = NdIndex::new(b.shape, b.strides);

        let mut a_n = 1;
        let mut b_n = 1;
        for i in ax..A::NUM_DIMS {
            a_n *= a_idx.shape[i];
            b_n *= b_idx.shape[i];
        }

        let mut i = 0;
        let n = grad_out.len();
        while i < n {
            for _ in 0..a_n {
                (*grad_a)[a_idx.next().unwrap()] += grad_out[i];
                i += 1;
            }
            for _ in 0..b_n {
                (*grad_b)[b_idx.next().unwrap()] += grad_out[i];
                i += 1;
            }
        }
        Ok(())
    }
}
