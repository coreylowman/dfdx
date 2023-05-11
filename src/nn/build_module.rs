use super::tensor_collection::*;
use crate::{
    shapes::{Dtype, Shape},
    tensor::{Storage, Tensor},
    tensor_ops::Device,
};

struct Builder<'a, E, D: Storage<E>> {
    device: &'a D,
    dtype: std::marker::PhantomData<E>,
}
impl<'a, E: Dtype, D: Device<E>> TensorVisitor<E, D> for Builder<'a, E, D> {
    type Viewer = ();
    type Err = D::Err;
    type E2 = E;
    type D2 = D;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        _t: (),
    ) -> Result<Option<Tensor<S, E, D>>, Self::Err> {
        let mut tensor: Tensor<S, E, D> = self.device.try_zeros_like(&opts.shape)?;
        (opts.reset)(&mut tensor)?;
        Ok(Some(tensor))
    }
}

/// Something that can be built. Related to [super::BuildOnDevice]
pub trait BuildModule<D: Device<E>, E: Dtype>:
    Sized + TensorCollection<E, D, To<E, D> = Self>
{
    /// Construct it on the device
    fn build(device: &D) -> Self {
        Self::try_build(device).unwrap()
    }

    /// Fallible version of [BuildModule::build]
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let out = Self::iter_tensors(&mut RecursiveWalker {
            m: (),
            f: &mut Builder {
                device,
                dtype: std::marker::PhantomData,
            },
        })?;

        Ok(out.unwrap())
    }
}

impl<D: Device<E>, E: Dtype, M: Sized + TensorCollection<E, D, To<E, D> = Self>> BuildModule<D, E>
    for M
{
}
