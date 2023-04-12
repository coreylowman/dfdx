use crate::{
    shapes::{Dtype, Shape},
    tensor::{unique_id, Cpu, Tensor},
};

impl<E: Dtype> super::ConcatKernel<E> for Cpu {
    fn forward<A: Shape, B: Shape>(
        &self,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
    ) -> Result<Tensor<A::Catted, E, Self>, Self::Err>
    where
        A: super::ConcatShape<B>,
    {
        let shape = a.shape.concat_shape(&b.shape);
        let mut data = self.try_alloc_elem(shape.num_elements(), E::default())?;
        let mut i = 0;
        if a.strides == a.shape.strides() {
            let a: &[E] = a.data.as_ref();
            data.data[0..a.len()].copy_from_slice(a);
            i += a.len();
        } else {
            let a_buf = a.as_vec();
            data.data[0..a_buf.len()].copy_from_slice(&a_buf);
            i += a_buf.len();
        }
        if b.strides == b.shape.strides() {
            let b: &[E] = b.data.as_ref();
            data.data[i..i + b.len()].copy_from_slice(b);
        } else {
            let b_buf = b.as_vec();
            data.data[i..i + b_buf.len()].copy_from_slice(&b_buf);
        }
        Ok(Tensor {
            id: unique_id(),
            data: std::sync::Arc::new(data),
            shape,
            strides: shape.strides(),
            device: self.clone(),
            tape: Default::default(),
        })
    }
    fn backward(
        &self,
        grad_a: &mut Self::Vec<E>,
        grad_b: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let mut offset = 0;
        for ga in grad_a.iter_mut() {
            *ga += grad_out[offset];
            offset += 1;
        }
        for gb in grad_b.iter_mut() {
            *gb += grad_out[offset];
            offset += 1;
        }
        Ok(())
    }
}
