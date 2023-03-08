use crate::{
    shapes::{Dtype, Shape},
    tensor::{Cpu, Tensor},
    unique_id::unique_id,
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
        let mut data = std::vec::Vec::with_capacity(shape.num_elements());
        if a.strides == a.shape.strides() {
            data.extend(a.data.as_ref());
        } else {
            data.extend(a.as_vec());
        }
        if b.strides == b.shape.strides() {
            data.extend(b.data.as_ref());
        } else {
            data.extend(b.as_vec());
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
    fn backward<A: Shape, B: Shape>(
        &self,
        _: &Tensor<A, E, Self>,
        grad_a: &mut Self::Vec<E>,
        _: &Tensor<B, E, Self>,
        grad_b: &mut Self::Vec<E>,
        _: &Tensor<A::Catted, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        A: super::ConcatShape<B>,
    {
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
