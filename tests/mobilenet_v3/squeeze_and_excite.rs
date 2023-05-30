use super::E;
use dfdx::prelude::*;

pub mod builder {
    use super::*;

    pub struct SqueezeAndExcite<const CHAN: usize, const CHAN_SQUEEZE: usize>;

    impl<const CHAN: usize, const CHAN_SQUEEZE: usize> BuildOnDevice<AutoDevice, E>
        for SqueezeAndExcite<CHAN, CHAN_SQUEEZE>
    where
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
    {
        type Built = super::SqueezeAndExcite<CHAN, CHAN_SQUEEZE, E, AutoDevice>;
    }
}

pub struct SqueezeAndExcite<const CHAN: usize, const CHAN_SQUEEZE: usize, E: Dtype, D: Device<E>> {
    fc1: modules::Conv2D<CHAN, CHAN_SQUEEZE, 1, 1, 0, 1, 1, E, D>,
    fc1_bias: modules::Bias2D<CHAN_SQUEEZE, E, D>,
    fc2: modules::Conv2D<CHAN_SQUEEZE, CHAN, 1, 1, 0, 1, 1, E, D>,
    fc2_bias: modules::Bias2D<CHAN, E, D>,
}

impl<const CHAN: usize, const CHAN_SQUEEZE: usize, E, D: Device<E>> TensorCollection<E, D>
    for SqueezeAndExcite<CHAN, CHAN_SQUEEZE, E, D>
where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
{
    type To<E2: Dtype, D2: Device<E2>> = SqueezeAndExcite<CHAN, CHAN_SQUEEZE, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("fc1", |s| &s.fc1, |s| &mut s.fc1),
                Self::module("fc1_bias", |s| &s.fc1_bias, |s| &mut s.fc1_bias),
                Self::module("fc2", |s| &s.fc2, |s| &mut s.fc2),
                Self::module("fc2_bias", |s| &s.fc2_bias, |s| &mut s.fc2_bias),
            ),
            |(fc1, fc1_bias, fc2, fc2_bias)| SqueezeAndExcite {
                fc1,
                fc1_bias,
                fc2,
                fc2_bias,
            },
        )
    }
}

impl<const CHAN: usize, const CHAN_SQUEEZE: usize, const WH: usize>
    Module<Tensor<Rank3<CHAN, WH, WH>, E, AutoDevice>>
    for SqueezeAndExcite<CHAN, CHAN_SQUEEZE, E, AutoDevice>
where
    Const<CHAN>: std::ops::Mul<Const<1>>,
    <Const<CHAN> as std::ops::Mul<Const<1>>>::Output: ConstDim + Dim,
    Const<CHAN_SQUEEZE>: std::ops::Mul<Const<1>>,
    <Const<CHAN_SQUEEZE> as std::ops::Mul<Const<1>>>::Output: ConstDim + Dim,
{
    type Output = Tensor<Rank3<CHAN, WH, WH>, E, AutoDevice>;
    type Error = <AutoDevice as HasErr>::Err;

    fn try_forward(
        &self,
        x: Tensor<Rank3<CHAN, WH, WH>, E, AutoDevice>,
    ) -> Result<Self::Output, Self::Error> {
        let inner_x: Tensor<Rank1<CHAN>, _, _, _> = x.with_empty_tape().try_mean()?;
        let inner_x = self
            .fc1_bias
            .try_forward(
                self.fc1
                    .try_forward(inner_x.try_reshape::<(_, Const<1>, Const<1>)>()?)?,
            )?
            .try_relu()?;
        let inner_x = self
            .fc2_bias
            .try_forward(
                self.fc2
                    .try_forward(inner_x.try_reshape::<(_, Const<1>, Const<1>)>()?)?,
            )?
            .try_hard_sigmoid()?;
        let y = inner_x
            .try_reshape::<Rank1<CHAN>>()?
            .try_broadcast::<Rank3<CHAN, WH, WH>, Axes2<1, 2>>()?;
        Ok(x * y)
    }
}

impl<const CHAN: usize, const CHAN_SQUEEZE: usize, const WH: usize, T>
    ModuleMut<Tensor<Rank3<CHAN, WH, WH>, E, AutoDevice, T>>
    for SqueezeAndExcite<CHAN, CHAN_SQUEEZE, E, AutoDevice>
where
    T: Tape<E, AutoDevice>,
    Const<CHAN>: std::ops::Mul<Const<1>>,
    <Const<CHAN> as std::ops::Mul<Const<1>>>::Output: ConstDim + Dim,
    Const<CHAN_SQUEEZE>: std::ops::Mul<Const<1>>,
    <Const<CHAN_SQUEEZE> as std::ops::Mul<Const<1>>>::Output: ConstDim + Dim,
{
    type Output = Tensor<Rank3<CHAN, WH, WH>, E, AutoDevice, T>;
    type Error = <AutoDevice as HasErr>::Err;

    fn try_forward_mut(
        &mut self,
        x: Tensor<Rank3<CHAN, WH, WH>, E, AutoDevice, T>,
    ) -> Result<Self::Output, Self::Error> {
        let inner_x: Tensor<Rank1<CHAN>, _, _, _> = x.with_empty_tape().try_mean()?;
        let inner_x = self
            .fc1_bias
            .try_forward_mut(
                self.fc1
                    .try_forward_mut(inner_x.try_reshape::<(_, Const<1>, Const<1>)>()?)?,
            )?
            .try_relu()?;
        let inner_x = self
            .fc2_bias
            .try_forward_mut(
                self.fc2
                    .try_forward_mut(inner_x.try_reshape::<(_, Const<1>, Const<1>)>()?)?,
            )?
            .try_hard_sigmoid()?;
        let y = inner_x
            .try_reshape::<Rank1<CHAN>>()?
            .try_broadcast::<Rank3<CHAN, WH, WH>, Axes2<1, 2>>()?;
        Ok(x * y)
    }
}

impl<const BATCH: usize, const CHAN: usize, const CHAN_SQUEEZE: usize, const WH: usize>
    Module<Tensor<Rank4<BATCH, CHAN, WH, WH>, E, AutoDevice>>
    for SqueezeAndExcite<CHAN, CHAN_SQUEEZE, E, AutoDevice>
where
    Const<CHAN>: std::ops::Mul<Const<1>>,
    <Const<CHAN> as std::ops::Mul<Const<1>>>::Output: ConstDim + Dim,
    Const<CHAN_SQUEEZE>: std::ops::Mul<Const<1>>,
    <Const<CHAN_SQUEEZE> as std::ops::Mul<Const<1>>>::Output: ConstDim + Dim,
{
    type Output = Tensor<Rank4<BATCH, CHAN, WH, WH>, E, AutoDevice>;
    type Error = <AutoDevice as HasErr>::Err;

    fn try_forward(
        &self,
        x: Tensor<Rank4<BATCH, CHAN, WH, WH>, E, AutoDevice>,
    ) -> Result<Self::Output, Self::Error> {
        let inner_x: Tensor<Rank2<BATCH, CHAN>, _, _, _> = x.with_empty_tape().try_mean()?;
        let inner_x = self
            .fc1_bias
            .try_forward(
                self.fc1
                    .try_forward(inner_x.try_reshape::<(Const<BATCH>, _, Const<1>, Const<1>)>()?)?,
            )?
            .try_relu()?;
        let inner_x = self
            .fc2_bias
            .try_forward(
                self.fc2
                    .try_forward(inner_x.try_reshape::<(Const<BATCH>, _, Const<1>, Const<1>)>()?)?,
            )?
            .try_hard_sigmoid()?;
        let y = inner_x
            .try_reshape::<Rank2<BATCH, CHAN>>()?
            .try_broadcast::<Rank4<BATCH, CHAN, WH, WH>, Axes2<2, 3>>()?;
        Ok(x * y)
    }
}

impl<const BATCH: usize, const CHAN: usize, const CHAN_SQUEEZE: usize, const WH: usize, T>
    ModuleMut<Tensor<Rank4<BATCH, CHAN, WH, WH>, E, AutoDevice, T>>
    for SqueezeAndExcite<CHAN, CHAN_SQUEEZE, E, AutoDevice>
where
    T: Tape<E, AutoDevice>,
    Const<CHAN>: std::ops::Mul<Const<1>>,
    <Const<CHAN> as std::ops::Mul<Const<1>>>::Output: ConstDim + Dim,
    Const<CHAN_SQUEEZE>: std::ops::Mul<Const<1>>,
    <Const<CHAN_SQUEEZE> as std::ops::Mul<Const<1>>>::Output: ConstDim + Dim,
{
    type Output = Tensor<Rank4<BATCH, CHAN, WH, WH>, E, AutoDevice, T>;
    type Error = <AutoDevice as HasErr>::Err;

    fn try_forward_mut(
        &mut self,
        x: Tensor<Rank4<BATCH, CHAN, WH, WH>, E, AutoDevice, T>,
    ) -> Result<Self::Output, Self::Error> {
        let inner_x: Tensor<Rank2<BATCH, CHAN>, _, _, _> = x.with_empty_tape().try_mean()?;
        let inner_x = self
            .fc1_bias
            .try_forward_mut(self.fc1.try_forward_mut(inner_x.try_reshape::<(
                Const<BATCH>,
                _,
                Const<1>,
                Const<1>,
            )>()?)?)?
            .try_relu()?;
        let inner_x = self
            .fc2_bias
            .try_forward_mut(self.fc2.try_forward_mut(inner_x.try_reshape::<(
                Const<BATCH>,
                _,
                Const<1>,
                Const<1>,
            )>()?)?)?
            .try_hard_sigmoid()?;
        let y = inner_x
            .try_reshape::<Rank2<BATCH, CHAN>>()?
            .try_broadcast::<Rank4<BATCH, CHAN, WH, WH>, Axes2<2, 3>>()?;
        Ok(x * y)
    }
}
