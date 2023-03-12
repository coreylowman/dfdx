use super::tensor_collection::*;
use crate::prelude::{Device, Dtype, HasShape, Shape, Tensor};

struct Converter<'a, D> {
    dev: &'a D,
}
impl<'a, E: Dtype, D: Device<E>, D2: Device<E>> TensorVisitor<E, D, E, D2> for Converter<'a, D2> {
    type Viewer = ViewTensorRef;
    type Err = D2::Err;

    fn visit<S: Shape>(
        &mut self,
        _opts: TensorOptions<S, E, D>,
        t: &Tensor<S, E, D>,
    ) -> Result<Option<Tensor<S, E, D2>>, Self::Err> {
        let buf = t.as_vec();
        Ok(Some(self.dev.try_tensor_from_vec(buf, *t.shape())?))
    }
}

/// Something that can be copied to another `Device`.
pub trait ToDevice<E: Dtype, D1: Device<E>, D2: Device<E>>: TensorCollection<E, D1> {
    /// Fallible version of [to_device]
    fn try_to_device(&self, device: &D2) -> Result<Self::Output<E, D2>, D2::Err> {
        let out = Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut Converter { dev: device },
        })?;
        Ok(out.unwrap())
    }

    /// Copy `self` from `D1` to `D2`
    fn to_device(&self, device: &D2) -> Self::Output<E, D2> {
        self.try_to_device(device).unwrap()
    }
}

impl<E: Dtype, D1: Device<E>, D2: Device<E>, T> ToDevice<E, D1, D2> for T where
    T: TensorCollection<E, D1>
{
}