use crate::shapes::*;
use crate::tensor::{Cpu, Tensor};

use std::sync::Arc;

use num_traits::Float;

use super::{Bilinear, NearestNeighbor};

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => panic!("Only implemented for 3d & 4d arrays"),
    }
}

impl<E: Float + Unit + std::ops::AddAssign + std::ops::DivAssign>
    super::Upscale2DKernel<E, NearestNeighbor> for Cpu
{
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let h_scale = ((istr[1] / istr[2]) as f32) / ((ostr[1] / ostr[2]) as f32);
        let w_scale = ((istr[2] / istr[3]) as f32) / ((ostr[2] / ostr[3]) as f32);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let ih = (h_scale * oh as f32) as usize;
                        let iw = (w_scale * ow as f32) as usize;
                        out_buf[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]] =
                            buf[b * istr[0] + c * istr[1] + ih * istr[2] + iw * istr[3]];
                    }
                }
            }
        }
        Ok(())
    }

    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Storage,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Storage,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let h_scale = ((istr[1] / istr[2]) as f32) / ((ostr[1] / ostr[2]) as f32);
        let w_scale = ((istr[2] / istr[3]) as f32) / ((ostr[2] / ostr[3]) as f32);

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let ih = (h_scale * oh as f32) as usize;
                        let iw = (w_scale * ow as f32) as usize;
                        grad_inp[b * istr[0] + c * istr[1] + ih * istr[2] + iw * istr[3]] +=
                            grad_out[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]];
                    }
                }
            }
        }
        Ok(())
    }
}

impl<E: Float + Unit + std::ops::AddAssign + std::ops::DivAssign>
    super::Upscale2DKernel<E, Bilinear> for Cpu
{
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let h_scale = ((istr[1] / istr[2] - 1) as f32) / ((ostr[1] / ostr[2] - 1) as f32);
        let w_scale = ((istr[2] / istr[3] - 1) as f32) / ((ostr[2] / ostr[3] - 1) as f32);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let ih = (h_scale * oh as f32) as usize;
                        let iw = (w_scale * ow as f32) as usize;

                        let hs = E::from(h_scale * oh as f32 - (ih as f32)).unwrap();
                        let ws = E::from(w_scale * ow as f32 - (iw as f32)).unwrap();

                        let one = E::from(1.0).unwrap();
                        let zero = E::from(0.0).unwrap();

                        let ll = buf[b * istr[0] + c * istr[1] + ih * istr[2] + iw * istr[3]]
                            * (one - hs)
                            * (one - ws);
                        let lh = if ws != zero {
                            buf[b * istr[0] + c * istr[1] + ih * istr[2] + (iw + 1) * istr[3]]
                                * (one - hs)
                                * ws
                        } else {
                            zero
                        };
                        let hl = if hs != zero {
                            buf[b * istr[0] + c * istr[1] + (ih + 1) * istr[2] + iw * istr[3]]
                                * hs
                                * (one - ws)
                        } else {
                            zero
                        };
                        let hh = if hs != zero && ws != zero {
                            buf[b * istr[0] + c * istr[1] + (ih + 1) * istr[2] + (iw + 1) * istr[3]]
                                * hs
                                * ws
                        } else {
                            zero
                        };

                        out_buf[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]] =
                            ll + lh + hl + hh;
                    }
                }
            }
        }
        Ok(())
    }

    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Storage,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Storage,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let h_scale = ((istr[1] / istr[2] - 1) as f32) / ((ostr[1] / ostr[2] - 1) as f32);
        let w_scale = ((istr[2] / istr[3] - 1) as f32) / ((ostr[2] / ostr[3] - 1) as f32);

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let ih = (h_scale * oh as f32) as usize;
                        let iw = (w_scale * ow as f32) as usize;

                        let hs = E::from(h_scale * oh as f32 - (ih as f32)).unwrap();
                        let ws = E::from(w_scale * ow as f32 - (iw as f32)).unwrap();

                        let one = E::from(1.0).unwrap();
                        let zero = E::from(0.0).unwrap();

                        let g = grad_out[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]];

                        grad_inp[b * istr[0] + c * istr[1] + ih * istr[2] + iw * istr[3]] +=
                            g * (one - hs) * (one - ws);
                        if ws != zero {
                            grad_inp
                                [b * istr[0] + c * istr[1] + ih * istr[2] + (iw + 1) * istr[3]] +=
                                g * (one - hs) * ws;
                        }
                        if hs != zero {
                            grad_inp
                                [b * istr[0] + c * istr[1] + (ih + 1) * istr[2] + iw * istr[3]] +=
                                g * hs * (one - ws);
                        }
                        if ws != zero && hs != zero {
                            grad_inp[b * istr[0]
                                + c * istr[1]
                                + (ih + 1) * istr[2]
                                + (iw + 1) * istr[3]] += g * hs * ws;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
