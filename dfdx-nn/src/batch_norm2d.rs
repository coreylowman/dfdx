use crate::{LoadSafeTensors, SaveSafeTensors, UpdateParams, ZeroGrads};
use dfdx::prelude::*;

#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct BatchNorm2DConfig<C: Dim>(pub C);

pub type BatchNorm2DConstConfig<const C: usize> = BatchNorm2DConfig<Const<C>>;

impl<C: Dim, E: Dtype, D: Device<E>> crate::BuildOnDevice<E, D> for BatchNorm2DConfig<C> {
    type Built = BatchNorm2D<C, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, D::Err> {
        Ok(BatchNorm2D {
            scale: device.try_ones_like(&(self.0,))?,
            bias: device.try_zeros_like(&(self.0,))?,
            running_mean: device.try_zeros_like(&(self.0,))?,
            running_var: device.try_ones_like(&(self.0,))?,
            epsilon: 1e-5,
            momentum: 0.1,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct BatchNorm2D<C: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[serialize]
    pub scale: Tensor<(C,), Elem, Dev>,
    #[param]
    #[serialize]
    pub bias: Tensor<(C,), Elem, Dev>,
    #[serialize]
    pub running_mean: Tensor<(C,), Elem, Dev>,
    #[serialize]
    pub running_var: Tensor<(C,), Elem, Dev>,
    #[serialize]
    pub epsilon: f64,
    #[serialize]
    pub momentum: f64,
}

impl<C: Dim, E: Dtype, D: Device<E>> crate::ResetParams<E, D> for BatchNorm2D<C, E, D> {
    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        self.scale.try_fill_with_ones()?;
        self.bias.try_fill_with_zeros()?;
        self.running_mean.try_fill_with_zeros()?;
        self.running_var.try_fill_with_ones()
    }
}

impl<C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(C, H, W), E, D, T>> for BatchNorm2D<C, E, D>
{
    type Output = Tensor<(C, H, W), E, D, T>;
    type Error = D::Err;
    fn try_forward(&self, x: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, Self::Error> {
        assert!(!T::OWNS_TAPE);
        self.infer_fwd(x)
    }
    fn try_forward_mut(
        &mut self,
        x: Tensor<(C, H, W), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        assert!(T::OWNS_TAPE);
        self.train_fwd(x)
    }
}

impl<Batch: Dim, C: Dim, H: Dim, W: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    crate::Module<Tensor<(Batch, C, H, W), E, D, T>> for BatchNorm2D<C, E, D>
{
    type Output = Tensor<(Batch, C, H, W), E, D, T>;
    type Error = D::Err;
    fn try_forward(
        &self,
        x: Tensor<(Batch, C, H, W), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        assert!(!T::OWNS_TAPE);
        self.infer_fwd(x)
    }
    fn try_forward_mut(
        &mut self,
        x: Tensor<(Batch, C, H, W), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        assert!(T::OWNS_TAPE);
        self.train_fwd(x)
    }
}

impl<C: Dim, E: Dtype, D: Device<E>> BatchNorm2D<C, E, D> {
    /// generic batchnorm forward for training
    fn train_fwd<S: Shape, T: Tape<E, D>, Ax: Axes>(
        &mut self,
        x: Tensor<S, E, D, T>,
    ) -> Result<Tensor<S, E, D, T>, D::Err>
    where
        S: HasAxes<Ax> + ReduceShapeTo<(C,), Ax>,
    {
        let n = <S as HasAxes<Ax>>::size(x.shape()) as f64;
        let shape = *x.shape();

        // compute statistics for updating running stats later - on tape
        let mean_chan = x.retaped::<T>().try_mean::<(C,), _>()?;

        // update statistics since we are training - off tape
        self.running_mean
            .try_axpy(1.0 - self.momentum, &mean_chan, self.momentum)?;

        let centered = x.try_sub(mean_chan.try_broadcast_like(&shape)?)?;

        let var_chan = centered
            .retaped::<T>()
            .try_square()?
            .try_mean::<(C,), _>()?;

        // NOTE: uses unbiased variance in running estimate
        self.running_var.try_axpy(
            1.0 - self.momentum,
            &var_chan,
            self.momentum * n / (n - 1.0),
        )?;

        // statistics for normalizing - on tape
        let std = var_chan
            .try_add(E::from_f64(self.epsilon).unwrap())?
            .try_sqrt()?;

        // record broadcast of scale & bias - on tape
        let scale = self
            .scale
            .retaped::<T>()
            .try_div(std)?
            .try_broadcast_like(&shape)?;
        let bias = self.bias.retaped::<T>().try_broadcast_like(&shape)?;

        // normalize & affine - on tape
        centered.try_mul(scale)?.try_add(bias)
    }

    /// generic batchnorm forward for inference
    pub fn infer_fwd<S: Shape, T: Tape<E, D>, Ax: Axes>(
        &self,
        x: Tensor<S, E, D, T>,
    ) -> Result<Tensor<S, E, D, T>, D::Err>
    where
        (C,): BroadcastShapeTo<S, Ax>,
    {
        let shape = *x.shape();

        // statistics for normalizing
        let std = self
            .running_var
            .clone()
            .try_add(E::from_f64(self.epsilon).unwrap())?
            .try_sqrt()?;

        let scale = self
            .scale
            .clone()
            .try_div(std)?
            .try_broadcast_like(&shape)?;

        // normalize & affine
        let x = x.try_sub(self.running_mean.clone().try_broadcast_like(&shape)?)?;
        let x = x.try_mul(scale)?;
        x.try_add(self.bias.clone().try_broadcast_like(&shape)?)
    }
}
