use crate::{
    shapes::{Dtype, Shape},
    tensor::{cpu::LendingIterator, Cpu, Tensor},
};

impl<E: Dtype> super::EmaKernel<E> for Cpu {
    fn forward<S: Shape>(
        &self,
        dst: &mut Tensor<S, E, Self>,
        src: &Tensor<S, E, Self>,
        decay: E,
    ) -> Result<(), Self::Err> {
        let mut dst_iter = dst.iter_mut();
        let mut src_iter = src.iter();
        while let Some((d, s)) = dst_iter.next().zip(src_iter.next()) {
            *d = *d * decay + *s * (E::ONE - decay);
        }
        Ok(())
    }
}
