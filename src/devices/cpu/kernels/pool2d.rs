use super::{device::Cpu, iterate::LendingIterator};
use crate::arrays::{Rank3, Shape};
use crate::devices::Device::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Standard;
use std::sync::Arc;

impl<
        const K: usize,
        const S: usize,
        const P: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >
    UnaryKernel<
        unary_ops::AvgPool2D<K, S, P>,
        Rank3<C, H, W>,
        Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
        f32,
    > for Cpu
{
    fn unary_fwd(
        &self,
        op: unary_ops::AvgPool2D<K, S, P>,
        inp: &mut Self::Storage<Rank3<C, H, W>, f32>,
    ) -> Result<
        Self::Storage<Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>, f32>,
        Self::Err,
    > {
        let mut out = self.try_const_zeros()?;
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        let inv_k2 = 1.0 / (K * K) as f32;
        let mut out_iter = out.iter_mut();
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut tmp = 0.0;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp += inp[[c, y, x]];
                                }
                            }
                        }
                    }
                    *out_iter.next().unwrap() = tmp * inv_k2;
                }
            }
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        op: unary_ops::AvgPool2D<K, S, P>,
        inp: &Self::Storage<Rank3<C, H, W>, f32>,
        grad_inp: &mut Self::Storage<Rank3<C, H, W>, f32>,
        grad_out: &Self::Storage<
            Rank3<C, { (H + 2 * P - K) / S + 1 }, { (W + 2 * P - K) / S + 1 }>,
            f32,
        >,
    ) {
        todo!();
    }
}
