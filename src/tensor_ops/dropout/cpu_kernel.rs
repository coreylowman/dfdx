use crate::{
    shapes::{Dtype, Shape},
    tensor::{unique_id, Cpu, Tensor},
};

use num_traits::Float;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Bernoulli, Distribution};

impl<E: Float + Dtype> super::DropoutKernel<E> for Cpu {
    fn forward<S: Shape>(
        &self,
        op: super::DropoutKernelOp,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        let dist = Bernoulli::new(op.prob).unwrap();
        let mut out = Tensor {
            id: unique_id(),
            data: inp.data.clone(),
            shape: inp.shape,
            strides: inp.strides,
            device: self.clone(),
            tape: Default::default(),
        };
        for x in out.buf_iter_mut() {
            *x = if dist.sample(&mut rng) {
                E::zero()
            } else {
                *x / E::from_f64(1.0 - op.prob).unwrap()
            };
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        op: super::DropoutKernelOp,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        let dist = Bernoulli::new(op.prob).unwrap();
        debug_assert_eq!(grad_inp.len(), grad_out.len());
        debug_assert_eq!(inp.data.len(), grad_out.len());
        for (i, data_i) in grad_inp.iter_mut().enumerate() {
            *data_i += if dist.sample(&mut rng) {
                E::zero()
            } else {
                E::from_f64((1.0 - op.prob).recip()).unwrap()
            } * grad_out[i];
        }
        Ok(())
    }
}
