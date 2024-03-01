mod tuples;
mod vecs;

use std::vec::Vec;

use crate::prelude::{Device, Dtype, Error, Gradients, Shape, Tensor, UniqueId};

/// Mutable & Immutable forward of `Input` that produces [Module::Output].
pub trait Module<X> {
    /// The type that this unit produces given `Input`.
    type Output;

    fn try_forward(&self, x: X) -> Result<Self::Output, Error>;

    fn try_forward_mut(&mut self, x: X) -> Result<Self::Output, Error> {
        self.try_forward(x)
    }

    fn forward(&self, x: X) -> Self::Output {
        self.try_forward(x).unwrap()
    }

    fn forward_mut(&mut self, x: X) -> Self::Output {
        self.try_forward_mut(x).unwrap()
    }
}

/// Something that can update both tensors and a [UpdateParams]. At minimum [Optimizer::update_tensor()] must be implemented.
pub trait Optimizer<M, E: Dtype, D: Device<E>>: Sized {
    fn update_tensor<S: Shape>(
        &mut self,
        t: &mut Tensor<S, E, D>,
        gradients: &Gradients<E, D>,
        missing_tensors: &mut Vec<UniqueId>,
    ) -> Result<(), Error>;

    fn update(&mut self, module: &mut M, gradients: &Gradients<E, D>) -> Result<(), Error>
    where
        M: UpdateParams<E, D>,
    {
        let mut missing_tensors = Vec::new();
        module.try_update_params(self, gradients, &mut missing_tensors)?;
        if missing_tensors.is_empty() {
            Ok(())
        } else {
            Err(Error::UnusedTensors(missing_tensors))
        }
    }
}

/// Something that can be constructed on a device as a certain dtype.
pub trait BuildOnDevice<E: Dtype, D: Device<E>>: Clone {
    type Built: Clone + std::fmt::Debug;
    fn build_on_device(&self, device: &D) -> Self::Built {
        self.try_build_on_device(device).unwrap()
    }
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error>;
}

/// Something that can have all of its parameters reset to a specific state (may be random or not random).
pub trait ResetParams<E: Dtype, D: Device<E>> {
    fn reset_params(&mut self) {
        self.try_reset_params().unwrap()
    }
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error>;
}

/// Something that can have it's params updated with an [Optimizer] and a set of [Gradients].
pub trait UpdateParams<E: Dtype, D: Device<E>> {
    fn update_params<M, Optim: Optimizer<M, E, D>>(
        &mut self,
        optimizer: &mut Optim,
        gradients: &Gradients<E, D>,
        missing_tensors: &mut Vec<UniqueId>,
    ) {
        self.try_update_params(optimizer, gradients, missing_tensors)
            .unwrap()
    }
    fn try_update_params<M, Optim: Optimizer<M, E, D>>(
        &mut self,
        optimizer: &mut Optim,
        gradients: &Gradients<E, D>,
        missing_tensors: &mut Vec<UniqueId>,
    ) -> Result<(), crate::tensor::Error>;
}

impl<S: Shape, E: Dtype, D: Device<E>> UpdateParams<E, D> for Tensor<S, E, D> {
    fn try_update_params<M, Optim: Optimizer<M, E, D>>(
        &mut self,
        optimizer: &mut Optim,
        gradients: &Gradients<E, D>,
        missing_tensors: &mut Vec<UniqueId>,
    ) -> Result<(), crate::tensor::Error> {
        optimizer.update_tensor(self, gradients, missing_tensors)
    }
}

/// Something that can allocate a [Gradients] object or zero out the [Gradients] object.
pub trait ZeroGrads<E: Dtype, D: Device<E>> {
    fn zero_grads(&self, grads: &mut Gradients<E, D>) {
        self.try_zero_grads(grads).unwrap()
    }
    fn try_zero_grads(&self, grads: &mut Gradients<E, D>) -> Result<(), crate::tensor::Error>;

    fn alloc_grads(&self) -> Gradients<E, D> {
        self.try_alloc_grads().unwrap()
    }
    fn try_alloc_grads(&self) -> Result<Gradients<E, D>, crate::tensor::Error> {
        let mut grads = Gradients::leaky();
        self.try_zero_grads(&mut grads)?;
        grads.retain_current_grads_as_leafs();
        Ok(grads)
    }
}

/// Something that can view or mutate a [Gradients] object.
pub trait WithGrads<E: Dtype, D: Device<E>> {
    /// View the gradient values for each parameter.
    fn grads_element_view<F: FnMut(&E)>(&self, grads: &Gradients<E, D>, f: F) {
        self.try_grads_element_view(grads, f).unwrap()
    }
    /// View the gradient values for each parameter.
    fn try_grads_element_view<F: FnMut(&E)>(
        &self,
        grads: &Gradients<E, D>,
        f: F,
    ) -> Result<(), Error>;
    /// View the gradient values for each tensor (unique id).
    fn grads_view<F: FnMut(&[E])>(&self, grads: &Gradients<E, D>, f: F) {
        self.try_grads_view(grads, f).unwrap()
    }
    /// View the gradient values for each tensor (unique id).
    fn try_grads_view<F: FnMut(&[E])>(&self, grads: &Gradients<E, D>, f: F) -> Result<(), Error>;
    /// Mutate the gradient values for each parameter.
    fn grads_element_map<F: FnMut(E) -> E>(&self, grads: &mut Gradients<E, D>, f: F) {
        self.try_grads_element_map(grads, f).unwrap()
    }
    /// Mutate the gradient values for each parameter.
    fn try_grads_element_map<F: FnMut(E) -> E>(
        &self,
        grads: &mut Gradients<E, D>,
        f: F,
    ) -> Result<(), crate::tensor::Error>;
    /// Mutate the gradient values for each tensor (unique id).
    fn grads_map<F: FnMut(Vec<E>) -> Option<Vec<E>>>(&self, grads: &mut Gradients<E, D>, f: F) {
        self.try_grads_map(grads, f).unwrap()
    }
    /// Mutate the gradient values for each tensor (unique id).
    fn try_grads_map<F: FnMut(Vec<E>) -> Option<Vec<E>>>(
        &self,
        grads: &mut Gradients<E, D>,
        f: F,
    ) -> Result<(), crate::tensor::Error>;
    /// Changes the gradient values for each parameter to be between `min` and `max`.
    ///
    /// Note that this may change the "direction" of your gradients.
    fn grads_clamp(&self, grads: &mut Gradients<E, D>, min: E, max: E)
    where
        E: std::cmp::PartialOrd + Clone,
    {
        self.try_grads_clamp(grads, min, max).unwrap()
    }
    /// Changes the gradient values for each parameter to be between `min` and `max`.
    ///
    /// Note that this may change the "direction" of your gradients.
    fn try_grads_clamp(&self, grads: &mut Gradients<E, D>, min: E, max: E) -> Result<(), Error>
    where
        E: std::cmp::PartialOrd + Clone,
    {
        self.try_grads_element_map(grads, |e| {
            if e < min {
                min
            } else if e > max {
                max
            } else {
                e
            }
        })
    }
    /// Changes the gradient values for each parameter to be between `-threshold` and `+threshold`.
    ///
    /// Note that this may change the "direction" of your gradients.
    fn grads_clip_value(&self, grads: &mut Gradients<E, D>, threshold: E)
    where
        E: std::cmp::PartialOrd + std::ops::Neg<Output = E> + Clone,
    {
        self.try_grads_clip_value(grads, threshold).unwrap()
    }
    /// Changes the gradient values for each parameter to be between `-threshold` and `+threshold`.
    ///
    /// Note that this may change the "direction" of your gradients.
    fn try_grads_clip_value(&self, grads: &mut Gradients<E, D>, threshold: E) -> Result<(), Error>
    where
        E: std::cmp::PartialOrd + std::ops::Neg<Output = E> + Clone,
    {
        self.try_grads_clamp(grads, -threshold, threshold)
    }
    /// Accumulates into `acc` the squared value for the gradients.
    ///
    /// After the accumulation, taking the sqrt of `acc` results in the gradients norm.
    fn grads_norm_squared(&self, grads: &Gradients<E, D>, acc: &mut E)
    where
        E: num_traits::Zero + std::ops::Mul<Output = E> + num_traits::Float,
    {
        self.try_grads_norm_squared(grads, acc).unwrap()
    }
    /// Accumulates into `acc` the squared value for the gradients.
    ///
    /// After the accumulation, taking the sqrt of `acc` results in the gradients norm.
    fn try_grads_norm_squared(&self, grads: &Gradients<E, D>, acc: &mut E) -> Result<(), Error>
    where
        E: std::ops::Mul<Output = E> + num_traits::Float,
    {
        self.try_grads_element_view(grads, |e| *acc += *e * *e)
    }
    /// Given a `norm` for all of the gradient values, scales down all gradients so their norm is not higher than `norm_threshold`.
    ///
    /// Note that this doesn't change the "direction" of your gradients.
    fn grads_clip_norm(&self, grads: &mut Gradients<E, D>, norm: E, norm_threshold: E)
    where
        E: Clone + std::cmp::PartialOrd + std::ops::Mul<Output = E> + std::ops::Div<Output = E>,
    {
        self.try_grads_clip_norm(grads, norm, norm_threshold)
            .unwrap()
    }
    /// Given a `norm` for all of the gradient values, scales down all gradients so their norm is not higher than `norm_threshold`.
    ///
    /// Note that this doesn't change the "direction" of your gradients.
    fn try_grads_clip_norm(
        &self,
        grads: &mut Gradients<E, D>,
        norm: E,
        norm_threshold: E,
    ) -> Result<(), Error>
    where
        E: Clone + std::cmp::PartialOrd + std::ops::Mul<Output = E> + std::ops::Div<Output = E>,
    {
        if norm > norm_threshold {
            self.try_grads_element_map(grads, |e| norm_threshold * e / norm)?
        }
        Ok(())
    }
}

#[cfg(feature = "safetensors")]
/// Something that can be saved to a .safetensors file.
pub trait SaveSafeTensors {
    fn save_safetensors<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), safetensors::SafeTensorError> {
        let mut tensors = Vec::new();
        self.write_safetensors("", &mut tensors);
        let data = tensors.iter().map(|(k, dtype, shape, data)| {
            (
                k.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        });

        safetensors::serialize_to_file(data, &None, path.as_ref())
    }
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    );
}

#[cfg(feature = "safetensors")]
/// Something that can be loaded from a .safetensors file.
pub trait LoadSafeTensors {
    fn load_safetensors<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), safetensors::SafeTensorError> {
        let f = std::fs::File::open(path)?;
        let buffer = unsafe { memmap2::MmapOptions::new().map(&f)? };
        let tensors = safetensors::SafeTensors::deserialize(&buffer)?;
        self.read_safetensors("", &tensors)
    }

    fn read_safetensors(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors,
    ) -> Result<(), safetensors::SafeTensorError>;
}

#[cfg(feature = "safetensors")]
impl<S: Shape, E: Dtype, D: Device<E>, T> LoadSafeTensors for Tensor<S, E, D, T> {
    fn read_safetensors(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors,
    ) -> Result<(), safetensors::SafeTensorError> {
        self.load_safetensor(tensors, location)
    }
}

#[cfg(feature = "safetensors")]
impl<S: Shape, E: Dtype, D: Device<E>, T> SaveSafeTensors for Tensor<S, E, D, T> {
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    ) {
        tensors.push((
            location.to_string(),
            <E as crate::dtypes::SafeTensorsDtype>::DTYPE,
            self.shape.concrete().into(),
            self.as_vec().iter().flat_map(|e| e.to_le_bytes()).collect(),
        ));
    }
}

macro_rules! unit_safetensors {
    ($Ty:ty) => {
        #[cfg(feature = "safetensors")]
        impl SaveSafeTensors for $Ty {
            fn write_safetensors(
                &self,
                location: &str,
                tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
            ) {
                #[allow(unused_imports)]
                use crate::dtypes::ToLeBytes;
                tensors.push((
                    location.to_string(),
                    <$Ty as crate::dtypes::SafeTensorsDtype>::DTYPE,
                    Vec::new(),
                    self.to_le_bytes().to_vec(),
                ));
            }
        }

        #[cfg(feature = "safetensors")]
        impl LoadSafeTensors for $Ty {
            fn read_safetensors(
                &mut self,
                location: &str,
                tensors: &safetensors::SafeTensors,
            ) -> Result<(), safetensors::SafeTensorError> {
                #[allow(unused_imports)]
                use crate::dtypes::FromLeBytes;
                let view = tensors.tensor(location)?;
                *self = Self::from_le_bytes(view.data().try_into().unwrap());
                Ok(())
            }
        }
    };
}

unit_safetensors!(bool);
unit_safetensors!(f32);
unit_safetensors!(f64);
unit_safetensors!(u8);
unit_safetensors!(u16);
unit_safetensors!(u32);
unit_safetensors!(u64);
unit_safetensors!(i8);
unit_safetensors!(i16);
unit_safetensors!(i32);
unit_safetensors!(i64);
unit_safetensors!(isize);
unit_safetensors!(usize);

/// Extension method that calls [BuildOnDevice] and then [ResetParams].
pub trait BuildModuleExt<M>: Sized {
    fn build_module<E: Dtype>(&self, m: M) -> M::Built
    where
        M: BuildOnDevice<E, Self>,
        M::Built: ResetParams<E, Self>,
        Self: Device<E>,
    {
        self.try_build_module(m).unwrap()
    }

    fn try_build_module<E: Dtype>(&self, m: M) -> Result<M::Built, Error>
    where
        M: BuildOnDevice<E, Self>,
        M::Built: ResetParams<E, Self>,
        Self: Device<E>,
    {
        let mut module = m.try_build_on_device(self)?;
        module.try_reset_params()?;
        Ok(module)
    }
}
impl<D, M> BuildModuleExt<M> for D {}
