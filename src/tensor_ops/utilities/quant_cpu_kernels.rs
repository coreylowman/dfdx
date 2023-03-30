use super::{
    cpu_kernels::{BinaryDerivative, UnaryDerivative},
    ops::{BinaryKernel, UnaryKernel},
};
use crate::{
    prelude::quant_cpu::Quantize,
    shapes::{Dtype, Shape},
    tensor::{cpu::LendingIterator, unique_id, QuantizedCpu, Tensor, ZerosTensor},
};

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync, Op: UnaryDerivative<K::Value>>
    UnaryKernel<Op, K::Value> for QuantizedCpu<K>
where
    K::Value: Dtype,
{
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: &Tensor<S, K::Value, Self>,
    ) -> Result<Tensor<S, K::Value, Self>, Self::Err> {
        let mut out = Tensor {
            id: unique_id(),
            data: inp.data.clone(),
            shape: inp.shape,
            strides: inp.strides,
            device: self.clone(),
            tape: Default::default(),
        };
        // NOTE: we can iterate over buf here because we know inp & out
        // have exact same strides due to clone.
        let mut iter = out.iter_blocks_mut();
        while let Some(mut block) = iter.next() {
            for x in block.iter_mut() {
                *x = op.f(x);
            }
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &Tensor<S, K::Value, Self>,
        grad_inp: &mut Self::Storage,
        grad_out: &Self::Storage,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_inp.len(), grad_out.len());
        debug_assert_eq!(inp.data.len(), grad_out.len());
        let mut iter = grad_inp.iter_blocks_mut();
        while let Some(mut block) = iter.next() {
            for (i, x) in block.iter_mut().enumerate() {
                *x += op.df(&inp.data.get(i).unwrap()) * grad_out.get(i).unwrap();
            }
        }
        Ok(())
    }
}

impl<K: 'static + Quantize + std::fmt::Debug + Send + Sync, Op: BinaryDerivative<K::Value>>
    BinaryKernel<Op, K::Value> for QuantizedCpu<K>
where
    K::Value: Dtype,
{
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: &Tensor<S, K::Value, Self>,
        rhs: &Tensor<S, K::Value, Self>,
    ) -> Result<Tensor<S, K::Value, Self>, Self::Err> {
        let mut out = self.try_zeros_like(&lhs.shape)?;

        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        let mut iter = out.iter_blocks_mut();
        while let Some(mut block) = iter.next() {
            for o in block.iter_mut() {
                let l = lhs_iter.next().unwrap();
                let r = rhs_iter.next().unwrap();
                *o = op.f(&l, &r);
            }
        }
        Ok(out)
    }
    fn backward<S: Shape>(
        &self,
        _op: Op,
        _lhs: &Tensor<S, K::Value, Self>,
        _grad_lhs: &mut Self::Storage,
        _rhs: &Tensor<S, K::Value, Self>,
        _grad_rhs: &mut Self::Storage,
        _grad_out: &Self::Storage,
    ) -> Result<(), Self::Err> {
        // let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
        // let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);

        // for go in grad_out.iter() {
        //     let lhs_i = lhs_idx.next().unwrap();
        //     let rhs_i = rhs_idx.next().unwrap();
        //     let l = lhs.data.get(lhs_i).unwrap();
        //     let r = rhs.data.get(rhs_i).unwrap();
        //     *grad_lhs.get_mut(lhs_i).unwrap() += op.dfdx(&l, &r) * go;
        //     *grad_rhs.get_mut(rhs_i).unwrap() += op.dfdy(&l, &r) * go;
        // }
        // Ok(())
        unimplemented!()
    }
}
