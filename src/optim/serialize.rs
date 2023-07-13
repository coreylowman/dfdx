use crate::{
    nn::tensor_collection::*,
    shapes::{Dtype, Shape},
    tensor::{Gradients, Tensor},
    tensor_ops::Device,
};

impl<'a, E: Dtype, D: Device<E>> TensorVisitor<E, D> for &'a Gradients<E, D> {
    type Viewer = ViewTensorRef;
    type Err = D::Err;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        t: &Tensor<S, E, D>,
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err> {
        let grad = if opts.do_gradient_update {
            self.get(t)
        } else {
            t.device.try_zeros_like(&t.shape)?
        };
        Ok(Some(grad))
    }
}

impl<E: Dtype, D: Device<E>> Gradients<E, D> {
    pub fn as_model<M>(&self, model: &M) -> Result<M, D::Err>
    where
        M: TensorCollection<E, D, To<E, D> = M>,
    {
        let mut op = self;
        Ok(M::iter_tensors(&mut RecursiveWalker {
            m: model,
            f: &mut op,
        })?
        .unwrap())
    }
}
