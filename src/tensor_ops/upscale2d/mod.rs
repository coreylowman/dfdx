mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::*,
    tensor::{DeviceStorage, HasErr, PutTape, SplitTape, Tape, Tensor, ZerosTensor},
};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Upscale2DOp {
    pub batch: usize,
    pub chan: usize,
    pub h_in: usize,
    pub h_out: usize,
    pub w_in: usize,
    pub w_out: usize,
}

impl Upscale2DOp {
    fn new([b, c, h_in, w_in]: [usize; 4], [h_out, w_out]: [usize; 2]) -> Self {
        assert!(
            h_out >= h_in,
            "Output height must be larger than input height"
        );
        assert!(
            w_out >= w_in,
            "Output width must be larger than input width"
        );
        Self {
            batch: b,
            chan: c,
            h_in,
            h_out,
            w_in,
            w_out,
        }
    }
}

pub trait UpscaleMethod: Default {}

#[derive(Clone, Copy, Default)]
pub struct NearestNeighbor;

impl UpscaleMethod for NearestNeighbor {}

#[derive(Clone, Copy, Default)]
pub struct Bilinear;

impl UpscaleMethod for Bilinear {}

pub trait Upscale2DKernel<E: Unit, M: UpscaleMethod>: DeviceStorage {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err>;

    fn backward<I: Shape, O: Shape>(
        &self,
        op: Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

pub trait ConstUpscale2D<M: UpscaleMethod>: HasErr {
    type Output<const OH: usize, const OW: usize>;
    fn try_upscale2d<const OH: usize, const OW: usize>(
        self,
    ) -> Result<Self::Output<OH, OW>, Self::Err>;
}

pub trait TryUpscale2D {
    fn upscale_2d<const OH: usize, const OW: usize, M: UpscaleMethod>(
        self,
    ) -> <Self as ConstUpscale2D<M>>::Output<OH, OW>
    where
        Self: ConstUpscale2D<M>,
    {
        self.try_upscale2d().unwrap()
    }
    fn try_upscale_2d<const OH: usize, const OW: usize, M: UpscaleMethod>(
        self,
    ) -> Result<<Self as ConstUpscale2D<M>>::Output<OH, OW>, Self::Err>
    where
        Self: ConstUpscale2D<M>,
    {
        ConstUpscale2D::try_upscale2d(self)
    }
}
impl<T> TryUpscale2D for T {}

impl<
        C: Dim,
        const H: usize,
        const W: usize,
        E: Dtype,
        M: UpscaleMethod,
        D: Upscale2DKernel<E, M> + ZerosTensor<E>,
        T: 'static + Tape<E, D>,
    > ConstUpscale2D<M> for Tensor<(C, Const<H>, Const<W>), E, D, T>
{
    type Output<const OH: usize, const OW: usize> = Tensor<(C, Const<OH>, Const<OW>), E, D, T>;

    fn try_upscale2d<const OH: usize, const OW: usize>(
        self,
    ) -> Result<Self::Output<OH, OW>, Self::Err> {
        let &(chan, _, _) = self.shape();
        let op = Upscale2DOp::new([1, chan.size(), H, W], [OH, OW]);
        let (inp, mut tape) = self.split_tape();
        let mut out = inp
            .device
            .try_zeros_like(&(chan, Default::default(), Default::default()))?;
        inp.device.forward(op, &inp, &mut out)?;
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device
                .backward(op, &inp, grad_inp, &phantom_out, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

impl<
        B: Dim,
        C: Dim,
        const H: usize,
        const W: usize,
        E: Dtype,
        M: UpscaleMethod,
        D: Upscale2DKernel<E, M> + ZerosTensor<E>,
        T: 'static + Tape<E, D>,
    > ConstUpscale2D<M> for Tensor<(B, C, Const<H>, Const<W>), E, D, T>
{
    type Output<const OH: usize, const OW: usize> = Tensor<(B, C, Const<OH>, Const<OW>), E, D, T>;

    fn try_upscale2d<const OH: usize, const OW: usize>(
        self,
    ) -> Result<Self::Output<OH, OW>, Self::Err> {
        let &(batch, chan, _, _) = self.shape();
        let op = Upscale2DOp::new([batch.size(), chan.size(), H, W], [OH, OW]);
        let (inp, mut tape) = self.split_tape();
        let mut out =
            inp.device
                .try_zeros_like(&(batch, chan, Default::default(), Default::default()))?;
        inp.device.forward(op, &inp, &mut out)?;
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device
                .backward(op, &inp, grad_inp, &phantom_out, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::*};

    use super::{NearestNeighbor, TryUpscale2D};

    #[test]
    fn nearest_upscale2d() {
        let dev = TestDevice::default();

        let x = dev.tensor([[[1.0, 0.0], [2.0, 3.0]]]);
        let y = x.leaky_trace().upscale_2d::<4, 4, NearestNeighbor>();
        assert_close(
            &y.array(),
            &[[
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 3.0, 3.0],
                [2.0, 2.0, 3.0, 3.0],
            ]],
        );

        let g = y.exp().mean().backward();
        assert_close(
            &g.get(&x).array(),
            &[[[0.679570457, 0.25], [1.847264025, 5.021384231]]],
        );
    }

    #[test]
    fn nearest_upscale2d_uneven() {
        let dev = TestDevice::default();

        let x = dev.tensor([[[1.0, 0.0, 2.0], [2.0, 3.0, 4.0]]]);
        let y = x.leaky_trace().upscale_2d::<2, 7, NearestNeighbor>();
        assert_close(
            &y.array(),
            &[[
                [1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            ]],
        );

        let g = y.exp().mean().backward();
        assert_close(
            &g.get(&x).array(),
            &[[
                [0.582488963, 0.142857143, 1.055579443],
                [1.583369164, 2.869362418, 7.799735719],
            ]],
        );
    }
}
