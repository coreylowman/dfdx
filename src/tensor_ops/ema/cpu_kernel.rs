use crate::{shapes::Dtype, tensor::Cpu};

impl<E: Dtype> super::EmaKernel<E> for Cpu {
    fn forward(
        &self,
        dst: &mut Self::Vec<E>,
        src: &Self::Vec<E>,
        decay: E,
    ) -> Result<(), Self::Err> {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *d * decay + *s * (E::ONE - decay);
        }
        Ok(())
    }
}
