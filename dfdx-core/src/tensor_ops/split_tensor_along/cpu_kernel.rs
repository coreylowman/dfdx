use super::AorB;
use crate::{
    shapes::*,
    tensor::{cpu::NdIndex, *},
};

impl<E: Dtype> super::SplitAlongKernel<E> for Cpu {
    fn forward<AB: Shape, A: Shape, B: Shape>(
        &self,
        ax: usize,
        ab: &Tensor<AB, E, Self>,
        a: &mut Tensor<A, E, Self>,
        b: &mut Tensor<B, E, Self>,
    ) -> Result<(), Error> {
        let mut a_n = 1;
        let mut b_n = 1;
        {
            let a_idx = NdIndex::new(a.shape, a.strides);
            let b_idx = NdIndex::new(b.shape, b.strides);
            for i in ax..A::NUM_DIMS {
                a_n *= a_idx.shape[i];
                b_n *= b_idx.shape[i];
            }
        }

        let n_ab = ab.data.len();

        let buf_a = std::sync::Arc::get_mut(&mut a.data).unwrap();
        let buf_b = std::sync::Arc::get_mut(&mut b.data).unwrap();

        let mut i = 0;
        let mut k = 0;
        let mut ab_idx = NdIndex::new(ab.shape, ab.strides);
        while i < n_ab {
            for j in 0..a_n {
                (*buf_a)[j + k * a_n] = ab.data[ab_idx.next().unwrap()];
                i += 1;
            }
            for j in 0..b_n {
                (*buf_b)[j + k * b_n] = ab.data[ab_idx.next().unwrap()];
                i += 1;
            }
            k += 1;
        }
        Ok(())
    }

    fn backward<AB: Shape, A: Shape, B: Shape>(
        &self,
        ax: usize,
        ab: &GhostTensor<AB, E, Self>,
        grad_ab: &mut Self::Vec,
        a: &GhostTensor<A, E, Self>,
        b: &GhostTensor<B, E, Self>,
        a_or_b: AorB,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let a_idx = NdIndex::new(a.shape, a.strides);
        let b_idx = NdIndex::new(b.shape, b.strides);

        let mut a_n = 1;
        let mut b_n = 1;
        for i in ax..A::NUM_DIMS {
            a_n *= a_idx.shape[i];
            b_n *= b_idx.shape[i];
        }

        let mut i = 0;
        let mut j = 0;
        let n = grad_ab.len();
        let mut ab_idx = NdIndex::new(ab.shape, ab.strides);
        while i + j < n {
            match a_or_b {
                AorB::A => {
                    for _ in 0..a_n {
                        (*grad_ab)[ab_idx.next().unwrap()] = grad_out[i];
                        i += 1;
                    }
                    for _ in 0..b_n {
                        ab_idx.next().unwrap();
                        j += 1;
                    }
                }
                AorB::B => {
                    for _ in 0..a_n {
                        ab_idx.next().unwrap();
                        j += 1;
                    }
                    for _ in 0..b_n {
                        (*grad_ab)[ab_idx.next().unwrap()] = grad_out[i];
                        i += 1;
                    }
                }
            };
        }

        Ok(())
    }
}
