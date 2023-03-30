use crate::{
    shapes::{Dtype, Shape},
    tensor::{cpu::LendingIterator, unique_id, Quantize, QuantizedCpu, Tensor},
};

use num_traits::Float;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Standard};

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync> super::DropoutKernel<K::Value>
    for QuantizedCpu<K>
where
    Standard: Distribution<K::Value>,
    K::Value: Dtype,
{
    fn forward<S: Shape>(
        &self,
        op: super::DropoutKernelOp<K::Value>,
        inp: &Tensor<S, K::Value, Self>,
    ) -> Result<Tensor<S, K::Value, Self>, Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        let mut out = Tensor {
            id: unique_id(),
            data: inp.data.clone(),
            shape: inp.shape,
            strides: inp.strides,
            device: self.clone(),
            tape: Default::default(),
        };
        let mut iter = out.iter_blocks_mut();
        while let Some(mut block) = iter.next() {
            for x in block.iter_mut() {
                let val: K::Value = rng.sample(Standard);
                *x = if val < op.prob {
                    <K::Value as num_traits::Zero>::zero()
                } else {
                    *x / (<K::Value as num_traits::One>::one() - op.prob)
                };
            }
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        op: super::DropoutKernelOp<K::Value>,
        inp: &Tensor<S, K::Value, Self>,
        grad_inp: &mut Self::Storage,
        grad_out: &Self::Storage,
    ) -> Result<(), Self::Err> {
        let mut rng = StdRng::seed_from_u64(op.seed);
        debug_assert_eq!(grad_inp.len(), grad_out.len());
        debug_assert_eq!(inp.data.len(), grad_out.len());
        let mut iter = grad_inp.iter_blocks_mut();
        let mut i = 0;
        while let Some(mut block) = iter.next() {
            for data_i in block.iter_mut() {
                let val: K::Value = rng.sample(Standard);
                *data_i += if val < op.prob {
                    <K::Value as num_traits::Zero>::zero()
                } else {
                    (<K::Value as num_traits::One>::one() - op.prob).recip()
                } * grad_out.get(i).unwrap();
                i += 1;
            }
        }
        Ok(())
    }
}
