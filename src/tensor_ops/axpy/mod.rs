use crate::{
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, Tensor},
};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

/// Elementwise `a * alpha + b * beta`.
///
/// See [Tensor::axpy] for in place version.
pub fn axpy<S: Shape, E: Dtype, D>(
    a: &Tensor<S, E, D>,
    alpha: impl Into<f64>,
    b: &Tensor<S, E, D>,
    beta: impl Into<f64>,
) -> Tensor<S, E, D>
where
    D: AxpyKernel<E>,
{
    let mut dst = a.clone();
    dst.axpy(alpha, b, beta);
    dst
}

impl<S: Shape, E: Dtype, D: AxpyKernel<E>> Tensor<S, E, D> {
    /// Updates self with elementwise function `self = self * alpha + b * beta`.
    pub fn axpy<T>(&mut self, alpha: impl Into<f64>, b: &Tensor<S, E, D, T>, beta: impl Into<f64>) {
        self.try_axpy(alpha, b, beta).unwrap()
    }

    /// Updates self with elementwise function `self = self * alpha + b * beta`.
    pub fn try_axpy<T>(
        &mut self,
        alpha: impl Into<f64>,
        b: &Tensor<S, E, D, T>,
        beta: impl Into<f64>,
    ) -> Result<(), D::Err> {
        assert_eq!(self.shape, b.shape);
        assert_eq!(self.strides, b.strides, "Strides must be equal for axpy");
        self.device.clone().forward(
            std::sync::Arc::make_mut(&mut self.data),
            E::from_f64(alpha.into()).unwrap(),
            b.data.as_ref(),
            E::from_f64(beta.into()).unwrap(),
        )
    }
}

pub trait AxpyKernel<E: Dtype>: DeviceStorage {
    fn forward(
        &self,
        a: &mut Self::Vec<E>,
        alpha: E,
        b: &Self::Vec<E>,
        beta: E,
    ) -> Result<(), Self::Err>;
}

#[cfg(test)]
mod tests {
    use crate::{shapes::Axis, tensor::*, tensor_ops::BroadcastTo, tests::*};

    #[test]
    #[should_panic = "left: `(5,)`,\n right: `(3,)`"]
    fn test_axpy_wrong_shape() {
        let dev: TestDevice = Default::default();
        let mut a: Tensor<_, TestDtype, _> = dev.zeros_like(&(5,));
        let b: Tensor<_, TestDtype, _> = dev.zeros_like(&(3,));
        a.axpy(0.99, &b, 0.01);
    }

    #[test]
    #[should_panic = "Strides must be equal for axpy"]
    fn test_axpy_wrong_strides() {
        let dev: TestDevice = Default::default();
        let mut a: Tensor<_, TestDtype, _> = dev.zeros_like(&(2, 5));
        let b: Tensor<_, TestDtype, _> =
            dev.zeros_like(&(5,)).broadcast_like::<_, Axis<0>>(&(2, 5));
        a.axpy(0.99, &b, 0.01);
    }

    #[test]
    fn test_axpy() {
        let dev: TestDevice = Default::default();

        let mut a = dev
            .tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]; 2])
            .to_dtype::<TestDtype>();
        let b = dev.tensor([[-1.5; 5], [1.5; 5]]).to_dtype::<TestDtype>();

        a.axpy(0.01, &b, 0.99);

        assert_close_to_literal!(
            a,
            [
                [-1.505, -1.495, -1.485, -1.475, -1.465],
                [1.465, 1.475, 1.485, 1.495, 1.505],
            ]
        );
    }
}
