use crate::{shapes::Dtype, tensor::Cpu};

impl<E: Dtype> super::AxpyKernel<E> for Cpu {
    fn forward(
        &self,
        a: &mut Self::Storage,
        alpha: E,
        b: &Self::Storage,
        beta: E,
    ) -> Result<(), Self::Err> {
        for (a_i, b_i) in a.iter_mut().zip(b.iter()) {
            *a_i = *a_i * alpha + *b_i * beta;
        }
        Ok(())
    }
}
