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
pub trait ToDtype<E1: Dtype, D: Device<E1>>: TensorCollection<E1, D> {
    /// Fallible version of [ToDtype::to_dtype]
    fn try_to_dtype<E2: Dtype>(&self) -> Result<Self::To<E2, D>, D::Err>
    where
        D: Device<E2> + ToDtypeKernel<E1, E2>,
    {
        let out = Self::iter_tensors(&mut RecursiveWalker {
            m: self,
            f: &mut Converter {
                e: Default::default(),
            },
        })?;
        Ok(out.unwrap())
    }

    /// Create a copy of `self` with dtype E2
    fn to_dtype<E2: Dtype>(&self) -> Self::To<E2, D>
    where
        D: Device<E2> + ToDtypeKernel<E1, E2>,
    {
        self.try_to_dtype().unwrap()
    }
}

impl<E1: Dtype, D: Device<E1>, T: TensorCollection<E1, D>> ToDtype<E1, D> for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::builders::*, tests::*};

    #[test]
    fn test_linear_f64_to_f32() {
        let dev: TestDevice = Default::default();
        type Model = Linear<2, 5>;
        let model_f64: modules::Linear<2, 5, f64, TestDevice> = dev.build_module::<Model, f64>();
        let model_f32: modules::Linear<2, 5, f32, TestDevice> = model_f64.to_dtype::<f32>();

        assert_eq!(
            model_f32.weight.as_vec(),
            model_f64
                .weight
                .as_vec()
                .iter()
                .map(|x| *x as f32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            model_f32.bias.as_vec(),
            model_f64
                .bias
                .as_vec()
                .iter()
                .map(|x| *x as f32)
                .collect::<Vec<_>>()
        );
    }
}
