use crate::{
    shapes::{Shape, Unit},
    tensor::{DeviceStorage, Tensor},
};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

/// Exponential moving average: `dst * decay + src * (1 - decay)`. See [ema_assign]
/// for in place version.
pub fn ema<S: Shape, E: Unit, D>(
    a: &Tensor<S, E, D>,
    b: &Tensor<S, E, D>,
    decay: E,
) -> Tensor<S, E, D>
where
    D: EmaKernel<E>,
{
    let mut dst = a.clone();
    dst.ema_assign(b, decay);
    dst
}

/// Exponential moving average: `dst = dst * decay + src * (1 - decay)`. See [ema] for non
/// in place version.
pub fn ema_assign<S: Shape, E: Unit, D>(dst: &mut Tensor<S, E, D>, src: &Tensor<S, E, D>, decay: E)
where
    D: EmaKernel<E>,
{
    dst.ema_assign(src, decay)
}

impl<S: Shape, E: Unit, D: EmaKernel<E>> Tensor<S, E, D> {
    /// See [ema] & [ema_assign]
    pub fn ema(&self, other: &Tensor<S, E, D>, decay: E) -> Tensor<S, E, D> {
        self.try_ema(other, decay).unwrap()
    }

    /// See [ema] & [ema_assign]
    pub fn try_ema(&self, other: &Tensor<S, E, D>, decay: E) -> Result<Tensor<S, E, D>, D::Err> {
        let mut dst = self.clone();
        dst.try_ema_assign(other, decay)?;
        Ok(dst)
    }

    /// See [ema] and [ema_assign]
    pub fn ema_assign(&mut self, src: &Tensor<S, E, D>, decay: E) {
        self.try_ema_assign(src, decay).unwrap()
    }

    /// See [ema] and [ema_assign]
    pub fn try_ema_assign(&mut self, src: &Tensor<S, E, D>, decay: E) -> Result<(), D::Err> {
        assert_eq!(self.shape, src.shape);
        assert_eq!(
            self.strides, src.strides,
            "Strides must be equal for in place op EMA"
        );
        self.device.clone().forward(
            std::sync::Arc::make_mut(&mut self.data),
            src.data.as_ref(),
            decay,
        )
    }
}

pub trait EmaKernel<E: Unit>: DeviceStorage {
    fn forward(
        &self,
        dst: &mut Self::Vec<E>,
        src: &Self::Vec<E>,
        decay: E,
    ) -> Result<(), Self::Err>;
}

#[cfg(test)]
mod tests {
    use crate::{shapes::Axis, tensor::*, tensor_ops::BroadcastTo, tests::*};

    #[test]
    #[should_panic = "left: `(5,)`,\n right: `(3,)`"]
    fn test_ema_wrong_shape() {
        let dev: TestDevice = Default::default();
        let mut a: Tensor<_, TestDtype, _> = dev.zeros_like(&(5,));
        let b: Tensor<_, TestDtype, _> = dev.zeros_like(&(3,));
        a.ema_assign(&b, 0.01);
    }

    #[test]
    #[should_panic = "Strides must be equal for in place op EMA"]
    fn test_ema_wrong_strides() {
        let dev: TestDevice = Default::default();
        let mut a: Tensor<_, TestDtype, _> = dev.zeros_like(&(2, 5));
        let b: Tensor<_, TestDtype, _> =
            dev.zeros_like(&(5,)).broadcast_like::<_, Axis<0>>(&(2, 5));
        a.ema_assign(&b, 0.01);
    }

    #[test]
    fn test_ema() {
        let dev: TestDevice = Default::default();

        let mut a: Tensor<_, TestDtype, _> = dev.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]; 2]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([[-1.5; 5], [1.5; 5]]);

        a.ema_assign(&b, 0.01);

        assert_close(
            &a.array(),
            &[
                [-1.505, -1.495, -1.485, -1.475, -1.465],
                [1.465, 1.475, 1.485, 1.495, 1.505],
            ],
        );
    }
}
