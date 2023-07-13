use crate::{
    shapes::{Dtype, Shape},
    tensor::{Gradients, Tensor},
    tensor_ops::Device,
    nn::tensor_collection::*,
};

impl<E: Dtype, D: Device<E>> TensorVisitor<E, D> for &Gradients<E, D> {
    type Viewer = ViewTensorRef;
    type Err = D::Err;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        t: &Tensor<S, E, D>,
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err> {
        if opts.do_gradient_update {
            Ok(Some(self.get(t)))
        } else {
            Ok(Some(t.device.zeros_like(&t.shape)))
        }
    }
}

impl<E: Dtype, D: Device<E>> TensorVisitor<E, D> for Gradients<E, D> {
    type Viewer = (ViewTensorRef, ViewTensorRef);
    type Err = D::Err;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        (grad, t): (&Tensor<S, E, D>, &Tensor<S, E, D>),
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err> {
        if opts.do_gradient_update {
            self.get_or_alloc_mut(t)?.clone_from(&grad.data);
        }
        Ok(None)
    }
}

impl<E: Dtype, D: Device<E>> Gradients<E, D> {
    /// Fallible version of [Gradients::to_model]
    pub fn try_to_model<Model: TensorCollection<E, D, To<E, D> = Model>>(
        &self,
        model: &Model,
    ) -> Result<Model, D::Err> {
        let mut f = self;
        let out = Model::iter_tensors(&mut RecursiveWalker {
            m: model,
            f: &mut f,
        })?;

        Ok(out.unwrap())
    }

    /// Creates a new model that contains the gradient values stored in `self`.
    ///
    /// # Panics
    /// This function panics if `self` does not contain a gradient for a trainable tensor in
    /// `model`.
    pub fn to_model<Model: TensorCollection<E, D, To<E, D> = Model>>(
        &self,
        model: &Model,
    ) -> Model {
        self.try_to_model(model).unwrap()
    }

    /// Fallible version of [Gradients::from_model]
    pub fn try_from_model<Model: TensorCollection<E, D>>(
        gradients: &Model,
        model: &Model,
    ) -> Result<Self, D::Err> {
        let mut out = Gradients::leaky();
        Model::iter_tensors(&mut RecursiveWalker {
            m: (gradients, model),
            f: &mut out,
        })?;
        Ok(out)
    }

    /// Creates new Gradients for `model` containing the gradients stored in `gradients`.
    pub fn from_model<Model: TensorCollection<E, D>>(gradients: &Model, model: &Model) -> Self {
        Self::try_from_model(gradients, model).unwrap()
    }
}

trait SerializeWithSurrogate<E: Dtype, D: Device<E>> {
    type Surrogate: TensorCollection<E, D>;

    fn get_surrogate(&self) -> Self::Surrogate;
    fn from_surrogate(surrogate: Self::Surrogate) -> Self;
}
