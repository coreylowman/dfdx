use super::tensor_collection::*;
use crate::{
    shapes::{Dtype, Shape},
    tensor::Tensor,
    tensor_ops::{Device, ToDtypeKernel},
};

struct Converter<E> {
    e: core::marker::PhantomData<E>,
}
impl<E1: Dtype, E2: Dtype, D: Device<E1> + Device<E2> + ToDtypeKernel<E1, E2>> TensorVisitor<E1, D>
    for Converter<E2>
{
    type Viewer = ViewTensorRef;
    type Err = D::Err;
    type E2 = E2;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        _opts: TensorOptions<S, E1, D>,
        t: &Tensor<S, E1, D>,
    ) -> Result<Option<Tensor<S, E2, D>>, Self::Err> {
        Ok(Some(t.clone().try_to_dtype()?))
    }
}

/// Something that can be copied to have a different dtype
pub trait ToDtype<E1: Dtype, E2: Dtype, D: Device<E1> + Device<E2> + ToDtypeKernel<E1, E2>>:
    TensorCollection<E1, D>
{
    /// Fallible version of [ToDevice::to_dtype]
    fn try_to_dtype(&self) -> Result<Self::To<E2, D>, D::Err> {
        let out = Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut Converter {
                e: Default::default(),
            },
        })?;
        Ok(out.unwrap())
    }

    /// Create a copy of `self` with dtype E2
    fn to_dtype(&self) -> Self::To<E2, D> {
        self.try_to_dtype().unwrap()
    }
}

impl<E1: Dtype, E2: Dtype, D: Device<E1> + Device<E2> + ToDtypeKernel<E1, E2>, T> ToDtype<E1, E2, D>
    for T
where
    T: TensorCollection<E1, D>,
{
}
