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

#[cfg(feature = "safetensors")]
/// Something that can be saved to a .safetensors file.
pub trait SaveSafeTensors {
    fn save_safetensors_with<P: AsRef<std::path::Path>, F: FnMut(String) -> String>(
        &self,
        path: P,
        key_map: &mut F,
    ) -> Result<(), safetensors::SafeTensorError> {
        let mut tensors = Vec::new();
        self.write_safetensors_with("", &mut tensors, key_map);
        let data = tensors.iter().map(|(k, dtype, shape, data)| {
            (
                k.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        });

        safetensors::serialize_to_file(data, &None, path.as_ref())
    }
    fn save_safetensors<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), safetensors::SafeTensorError> {
        self.save_safetensors_with(path, &mut core::convert::identity)
    }
    fn write_safetensors_with<F: FnMut(String) -> String>(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
        key_map: &mut F,
    );
    fn write_safetensors(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    ) {
        self.write_safetensors_with(location, tensors, &mut core::convert::identity)
    }
}

#[cfg(feature = "safetensors")]
/// Something that can be loaded from a .safetensors file.
pub trait LoadSafeTensors {
    fn load_safetensors_with<P: AsRef<std::path::Path>, F: FnMut(String) -> String>(
        &mut self,
        path: P,
        skip_missing: bool,
        key_map: &mut F,
    ) -> Result<(), safetensors::SafeTensorError> {
        let f = std::fs::File::open(path)?;
        let buffer = unsafe { memmap2::MmapOptions::new().map(&f)? };
        let tensors = safetensors::SafeTensors::deserialize(&buffer)?;
        self.read_safetensors_with("", &tensors, skip_missing, key_map)
    }
    fn load_safetensors<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), safetensors::SafeTensorError> {
        self.load_safetensors_with(path, false, &mut core::convert::identity)
    }
    fn load_safetensors_from_bytes_with<F: FnMut(String) -> String>(
        &mut self,
        bytes: &[u8],
        skip_missing: bool,
        key_map: &mut F,
    ) -> Result<(), safetensors::SafeTensorError> {
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        self.read_safetensors_with("", &tensors, skip_missing, key_map)
    }
    fn load_safetensors_from_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), safetensors::SafeTensorError> {
        self.load_safetensors_from_bytes_with(bytes, false, &mut core::convert::identity)
    }

    fn read_safetensors_with<F: FnMut(String) -> String>(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors,
        skip_missing: bool,
        key_map: &mut F,
    ) -> Result<(), safetensors::SafeTensorError>;
    fn read_safetensors(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors,
    ) -> Result<(), safetensors::SafeTensorError> {
        self.read_safetensors_with(location, tensors, false, &mut core::convert::identity)
    }
}

#[cfg(feature = "safetensors")]
impl<S: Shape, E: Dtype, D: Device<E>, T> LoadSafeTensors for Tensor<S, E, D, T> {
    fn read_safetensors_with<F: FnMut(String) -> String>(
        &mut self,
        location: &str,
        tensors: &safetensors::SafeTensors,
        skip_missing: bool,
        key_map: &mut F,
    ) -> Result<(), safetensors::SafeTensorError> {
        self.load_safetensor(tensors, location, skip_missing, key_map)
    }
}

#[cfg(feature = "safetensors")]
impl<S: Shape, E: Dtype, D: Device<E>, T> SaveSafeTensors for Tensor<S, E, D, T> {
    fn write_safetensors_with<F: FnMut(String) -> String>(
        &self,
        location: &str,
        tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
        key_map: &mut F,
    ) {
        let location = key_map(location.to_string());
        tensors.push((
            location,
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
            fn write_safetensors_with<F: FnMut(String) -> String>(
                &self,
                location: &str,
                tensors: &mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
                key_map: &mut F,
            ) {
                let location = key_map(location.to_string());
                #[allow(unused_imports)]
                use crate::dtypes::ToLeBytes;
                tensors.push((
                    location,
                    <$Ty as crate::dtypes::SafeTensorsDtype>::DTYPE,
                    Vec::new(),
                    self.to_le_bytes().to_vec(),
                ));
            }
        }

        #[cfg(feature = "safetensors")]
        impl LoadSafeTensors for $Ty {
            fn read_safetensors_with<F: FnMut(String) -> String>(
                &mut self,
                location: &str,
                tensors: &safetensors::SafeTensors,
                skip_missing: bool,
                key_map: &mut F,
            ) -> Result<(), safetensors::SafeTensorError> {
                let location = key_map(location.to_string());
                #[allow(unused_imports)]
                use crate::dtypes::FromLeBytes;
                let view = match tensors.tensor(&location) {
                    Ok(ok) => ok,
                    Err(safetensors::SafeTensorError::TensorNotFound(_name)) if skip_missing => {
                        return Ok(());
                    }
                    Err(e) => return Err(e),
                };
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
