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
        assert!(h_out>=h_in, "Output height must be larger than input height");
        assert!(w_out>=w_in, "Output width must be larger than input width");
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

macro_rules! upscale2d {
    (Kernel=$Kernel:ident, ConstTrait=$ConstTrait:ident, TryTrait=$TryTrait:ident, Meth=$Meth:ident, TryMeth=$TryMeth:ident) => {
        pub trait $Kernel<E: Unit>: DeviceStorage {
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

        pub trait $ConstTrait<const OH: usize, const OW: usize>: HasErr {
            type Output;
            fn try_upscale2d(self) -> Result<Self::Output, Self::Err>;
        }

        pub trait $TryTrait {
            fn $Meth<const OH: usize, const OW: usize>(self) -> Self::Output
            where
                Self: $ConstTrait<OH, OW>,
            {
                self.try_upscale2d().unwrap()
            }
            fn $TryMeth<const OH: usize, const OW: usize>(
                self,
            ) -> Result<Self::Output, Self::Err>
            where
                Self: $ConstTrait<OH, OW>,
            {
                self.try_upscale2d()
            }
        }
        impl<T> $TryTrait for T {}

        impl<
                C: Dim,
                const H: usize,
                const W: usize,
                const OH: usize,
                const OW: usize,
                E: Dtype,
                D: $Kernel<E> + ZerosTensor<E>,
                T: 'static + Tape<E, D>,
            > $ConstTrait<OH, OW> for Tensor<(C, Const<H>, Const<W>), E, D, T>
        {
            type Output = Tensor<
                (
                    C,
                    Const<OH>,
                    Const<OW>,
                ),
                E,
                D,
                T,
            >;

            fn try_upscale2d(self) -> Result<Self::Output, Self::Err> {
                let &(chan, _, _) = self.shape();
                let op = Upscale2DOp::new([1, chan.size(), H, W], [OH, OW]);
                let (inp, mut tape) = self.split_tape();
                let mut out =
                    inp.device
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
                const OH: usize,
                const OW: usize,
                E: Dtype,
                D: $Kernel<E> + ZerosTensor<E>,
                T: 'static + Tape<E, D>,
            > $ConstTrait<OH, OW> for Tensor<(B, C, Const<H>, Const<W>), E, D, T>
        where
        {
            type Output = Tensor<
                (
                    B,
                    C,
                    Const<OH>,
                    Const<OW>,
                ),
                E,
                D,
                T,
            >;

            fn try_upscale2d(self) -> Result<Self::Output, Self::Err> {
                let &(batch, chan, _, _) = self.shape();
                let op = Upscale2DOp::new([batch.size(), chan.size(), H, W], [OH, OW]);
                let (inp, mut tape) = self.split_tape();
                let mut out = inp.device.try_zeros_like(&(
                    batch,
                    chan,
                    Default::default(),
                    Default::default(),
                ))?;
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
    };
}

upscale2d!(
    Kernel = NearestUpscale2DKernel,
    ConstTrait = ConstNearestUpscale2D,
    TryTrait = TryNearestUpscale2D,
    Meth = nearest_upscale2d,
    TryMeth = try_nearest_upscale2d
);

#[cfg(test)]
mod tests {
    use crate::{tests::*, prelude::*};

    use super::TryNearestUpscale2D;
    
    #[test]
    fn nearest_upscale2d() {
        let dev = TestDevice::default();

        let x = dev.tensor([[[1.0, 0.0],[2.0, 3.0]]]);
        let y = x.leaky_trace().nearest_upscale2d::<4, 4>();
        assert_close(&y.array(), &[[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0]]]);

        let g = y.exp().mean().backward();
        assert_close(&g.get(&x).array(), &[[[0.679570457, 0.25], [1.847264025, 5.021384231]]]);
    }

    #[test]
    fn nearest_upscale2d_uneven() {
        let dev = TestDevice::default();

        let x = dev.tensor([[[1.0, 0.0, 2.0],[2.0, 3.0, 4.0]]]);
        let y = x.leaky_trace().nearest_upscale2d::<2, 7>();
        assert_close(&y.array(), &[[[1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]]);

        let g = y.exp().mean().backward();
        assert_close(&g.get(&x).array(), &[[[0.582488963, 0.142857143, 1.055579443], 
                                                 [1.583369164, 2.869362418, 7.799735719]]]);
    }
}
