use crate::{
    shapes::{Dtype, Shape},
    tensor::Cpu,
};

use num_traits::Float;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Standard};

impl<F: Float + Dtype> super::DropoutKernel<F> for Cpu
where
    Standard: Distribution<F>,
{
    fn forward<S: Shape>(
        &self,
        op: super::DropoutKernelOp<F>,
        inp: &Self::Storage<S, F>,
    ) -> Result<Self::Storage<S, F>, Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        let mut out: Self::Storage<S, F> = inp.clone();
        for x in out.buf_iter_mut() {
            let val: F = rng.sample(Standard);
            *x = if val < op.prob {
                F::zero()
            } else {
                *x / (F::one() - op.prob)
            };
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        op: super::DropoutKernelOp<F>,
        inp: &Self::Storage<S, F>,
        grad_inp: &mut Self::Storage<S, F>,
        grad_out: &Self::Storage<S, F>,
    ) -> Result<(), Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        debug_assert_eq!(inp.data.len(), grad_out.data.len());
        for (i, data_i) in grad_inp.buf_iter_mut().enumerate() {
            let val: F = rng.sample(Standard);
            *data_i += if val < op.prob {
                F::zero()
            } else {
                (F::one() - op.prob).recip()
            } * grad_out.data[i];
        }
        Ok(())
    }
}
