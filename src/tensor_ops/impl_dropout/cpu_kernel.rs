use super::DropoutKernelOp;
use crate::{
    arrays::Shape,
    devices::{Cpu, UnaryKernel},
};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Standard;

impl<S: Shape> UnaryKernel<DropoutKernelOp, S, S, f32> for Cpu {
    fn unary_fwd(
        &self,
        op: DropoutKernelOp,
        inp: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, f32>, Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        let mut out: Self::Storage<S, f32> = inp.try_clone()?;
        for x in out.buf_iter_mut() {
            let val: f32 = rng.sample(Standard);
            *x = if val < op.prob {
                0.0
            } else {
                *x / (1.0 - op.prob)
            };
        }
        Ok(out)
    }

    fn unary_bwd(
        &self,
        op: DropoutKernelOp,
        inp: &Self::Storage<S, f32>,
        grad_inp: &mut Self::Storage<S, f32>,
        grad_out: &Self::Storage<S, f32>,
    ) -> Result<(), Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        assert_eq!(grad_inp.data.len(), grad_out.data.len());
        assert_eq!(inp.data.len(), grad_out.data.len());
        for (i, data_i) in grad_inp.buf_iter_mut().enumerate() {
            let val: f32 = rng.sample(Standard);
            *data_i += if val < op.prob {
                0.0
            } else {
                1.0 / (1.0 - op.prob)
            } * grad_out.data[i];
        }
        Ok(())
    }
}
