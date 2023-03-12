use super::tensor_collection::*;
use crate::prelude::{Device, DeviceStorage, Dtype, Shape, Tensor};

struct Builder<'a, D: DeviceStorage>(&'a D);
impl<'a, E: Dtype, D: Device<E>> TensorVisitor<E, D, E, D> for Builder<'a, D> {
    type Viewer = ();
    type Err = D::Err;

    fn visit<S: Shape>(
        &mut self,
        opts: TensorOptions<S, E, D>,
        _t: (),
    ) -> TensorVisitorOutput<Self, S, E, D, E, D> {
        let mut tensor: Tensor<S, E, D> = self.0.try_zeros_like(&opts.shape)?;
        (opts.reset)(&mut tensor)?;
        Ok(Some(tensor))
    }
}

/// Something that can be built. Related to [BuildOnDevice]
pub trait BuildModule<D: Device<E>, E: Dtype>:
    Sized + TensorCollection<E, D, Output<E, D> = Self>
{
    /// Construct it on the device
    fn build(device: &D) -> Self {
        Self::try_build(device).unwrap()
    }

    /// Fallible version of [BuildModule::build]
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let out = Self::iter_tensors(&mut RecursiveWalker {
            m: (),
            f: &mut Builder(device),
        })?;

        Ok(out.unwrap())
    }
}

impl<D: Device<E>, E: Dtype, M: Sized + TensorCollection<E, D, Output<E, D> = Self>>
    BuildModule<D, E> for M
{
}
