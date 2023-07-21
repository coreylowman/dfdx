#[cfg(any(feature = "safetensors", feature = "numpy"))]
use std::path::Path;

use crate::{
    nn::tensor_collection::*,
    shapes::{Dtype, Shape},
    tensor::{Gradients, Tensor},
    tensor_ops::Device,
};

#[cfg(feature = "numpy")]
use crate::{
    nn::{LoadFromNpz, SaveToNpz},
    tensor::{numpy::NpzError, NumpyDtype},
};

#[cfg(feature = "numpy")]
use zip::{result::ZipResult, ZipArchive, ZipWriter};

#[cfg(feature = "safetensors")]
use crate::{
    nn::{LoadFromSafetensors, SaveToSafetensors},
    tensor::safetensors::SafeDtype,
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

impl<M, E: Dtype, D: Device<E>> SerializeWithModel<M, E, D> for Gradients<E, D>
where
    M: TensorCollection<E, D, To<E, D> = M> + Clone,
{
    type Serializer = M;

    fn try_to_serializer(&self, model: &M) -> Result<M, <D>::Err> {
        let mut f = self;
        let out = M::iter_tensors(&mut RecursiveWalker {
            m: model,
            f: &mut f,
        })?;

        Ok(out.unwrap())
    }

    fn try_from_serializer(data: &M, model: &M) -> Result<Self, <D>::Err> {
        let mut out = Gradients::leaky();
        M::iter_tensors(&mut RecursiveWalker {
            m: (data, model),
            f: &mut out,
        })?;
        Ok(out)
    }
}

pub trait SerializeWithModel<M, E: Dtype, D: Device<E>>: Sized
where
    M: TensorCollection<E, D> + Clone,
{
    type Serializer: TensorCollection<E, D>;

    /// Fallible version of [Gradients::to_model]
    fn try_to_serializer(&self, model: &M) -> Result<Self::Serializer, D::Err>;

    /// Fallible version of [Gradients::from_model]
    fn try_from_serializer(serializer: &Self::Serializer, model: &M) -> Result<Self, D::Err>;

    /// Converts the data of `self` to the structure of `model` so that its data can be serialized.
    ///
    /// # Panics
    /// This function may panic if `self` does not contain a tensor corresponding to a trainable
    /// tensor in `model`.
    fn to_serializer(&self, model: &M) -> Self::Serializer {
        self.try_to_serializer(model).unwrap()
    }

    /// Creates an instance of `Self` containing the tensors in `data`, where each tensor in `self`
    /// is associated with a corresponding tensor in `model`.
    fn from_serializer(serializer: &Self::Serializer, model: &M) -> Self {
        Self::try_from_serializer(serializer, model).unwrap()
    }

    /// See [SaveToNpz::save].
    #[cfg(feature = "numpy")]
    fn save<P: AsRef<Path>>(&self, path: P, model: &M) -> ZipResult<()>
    where
        E: NumpyDtype,
    {
        self.to_serializer(model).save(path)
    }

    /// See [SaveToNpz::write].
    #[cfg(feature = "numpy")]
    fn write<W>(&self, w: &mut ZipWriter<W>, model: &M) -> ZipResult<()>
    where
        W: std::io::Write + std::io::Seek,
        E: NumpyDtype,
    {
        self.to_serializer(model).write(w)
    }

    /// See [LoadFromNpz::load].
    #[cfg(feature = "numpy")]
    fn load<P: AsRef<Path>>(&mut self, path: P, model: &M) -> Result<(), NpzError>
    where
        E: NumpyDtype,
    {
        let mut serializer = self.to_serializer(model);
        serializer.load(path)?;
        *self = Self::from_serializer(&serializer, model);
        Ok(())
    }

    /// See [LoadFromNpz::read].
    #[cfg(feature = "numpy")]
    fn read<R>(&mut self, r: &mut ZipArchive<R>, model: &M) -> Result<(), NpzError>
    where
        R: std::io::Read + std::io::Seek,
        E: NumpyDtype,
    {
        let mut serializer = self.to_serializer(model);
        serializer.read(r)?;
        *self = Self::from_serializer(&serializer, model);
        Ok(())
    }

    /// See [SaveToSafetensors::save_safetensors].
    #[cfg(feature = "safetensors")]
    fn save_safetensors<P: AsRef<Path>>(
        &self,
        path: P,
        model: &M,
    ) -> Result<(), safetensors::SafeTensorError>
    where
        E: SafeDtype,
    {
        self.to_serializer(model).save_safetensors(path)
    }

    /// See [LoadFromSafetensors::load_safetensors].
    #[cfg(feature = "safetensors")]
    fn load_safetensors<P: AsRef<Path>>(
        &mut self,
        path: P,
        model: &M,
    ) -> Result<(), crate::tensor::safetensors::Error>
    where
        E: SafeDtype,
    {
        let mut serializer = self.to_serializer(model);
        serializer.load_safetensors(path)?;
        *self = Self::from_serializer(&serializer, model);
        Ok(())
    }
}
