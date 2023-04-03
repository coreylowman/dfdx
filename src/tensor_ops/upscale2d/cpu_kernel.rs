use crate::shapes::*;
use crate::tensor::{Cpu, Tensor};

use std::sync::Arc;

use num_traits::Float;

use super::{Bilinear, NearestNeighbor};

fn make_4d<S: Shape>(items: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, items[0], items[1], items[2]],
        4 => [items[0], items[1], items[2], items[3]],
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

        let y_ratio = (op.h_in as f32) / (op.h_out as f32);
        let x_ratio = (op.w_in as f32) / (op.w_out as f32);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for y_out in 0..op.h_out {
                    for x_out in 0..op.w_out {
                        let y_in = (y_ratio * y_out as f32).floor() as usize;
                        let x_in = (x_ratio * x_out as f32).floor() as usize;
                        let y_in = y_in.min(op.h_in - 1);
                        let x_in = x_in.min(op.w_in - 1);
                        out_buf[b * ostr[0] + c * ostr[1] + y_out * ostr[2] + x_out * ostr[3]] =
                            buf[b * istr[0] + c * istr[1] + y_in * istr[2] + x_in * istr[3]];
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
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let y_ratio = (op.h_in as f32) / (op.h_out as f32);
        let x_ratio = (op.w_in as f32) / (op.w_out as f32);

        for b in 0..op.batch {
            for c in 0..op.chan {
                for y_out in 0..op.h_out {
                    for x_out in 0..op.w_out {
                        let y_in: usize = (y_ratio * y_out as f32).floor() as usize;
                        let y_in = y_in.min(op.h_in - 1);
                        let x_in: usize = (x_ratio * x_out as f32).floor() as usize;
                        let x_in = x_in.min(op.w_in - 1);
                        grad_inp[b * istr[0] + c * istr[1] + y_in * istr[2] + x_in * istr[3]] +=
                            grad_out[b * ostr[0] + c * ostr[1] + y_out * ostr[2] + x_out * ostr[3]];
                    }
                }
            }
        }
        Ok(())
    }
}

impl<E: Float + Dtype> super::Upscale2DKernel<E, Bilinear> for Cpu {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let y_ratio = ((op.h_in - 1) as f32) / ((op.h_out - 1) as f32);
        let x_ratio = ((op.w_in - 1) as f32) / ((op.w_out - 1) as f32);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for y_out in 0..op.h_out {
                    for x_out in 0..op.w_out {
                        let x_frac = x_ratio * x_out as f32;
                        let x0 = x_frac.floor().min((op.w_in - 1) as f32);
                        let x1 = x_frac.ceil().min((op.w_in - 1) as f32);
                        let xw = E::from_f32(x_frac - x0).unwrap();

                        let y_frac = y_ratio * y_out as f32;
                        let y0 = y_frac.floor().min((op.h_in - 1) as f32);
                        let y1 = y_frac.ceil().min((op.h_in - 1) as f32);
                        let yw = E::from_f32(y_frac - y0).unwrap();

                        let [x0, x1, y0, y1] = [x0, x1, y0, y1].map(|q| q as usize);
                        debug_assert!(x0 < op.w_in);
                        debug_assert!(x1 < op.w_in);
                        debug_assert!(y0 < op.h_in);
                        debug_assert!(y1 < op.h_in);

                        let p_a = buf[b * istr[0] + c * istr[1] + y0 * istr[2] + x0 * istr[3]];
                        let p_b = buf[b * istr[0] + c * istr[1] + y0 * istr[2] + x1 * istr[3]];
                        let p_c = buf[b * istr[0] + c * istr[1] + y1 * istr[2] + x0 * istr[3]];
                        let p_d = buf[b * istr[0] + c * istr[1] + y1 * istr[2] + x1 * istr[3]];

                        let p_a = p_a * (E::ONE - xw) * (E::ONE - yw);
                        let p_b = p_b * xw * (E::ONE - yw);
                        let p_c = p_c * (E::ONE - xw) * yw;
                        let p_d = p_d * xw * yw;

                        out_buf[b * ostr[0] + c * ostr[1] + y_out * ostr[2] + x_out * ostr[3]] =
                            p_a + p_b + p_c + p_d;
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
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let y_ratio = ((op.h_in - 1) as f32) / ((op.h_out - 1) as f32);
        let x_ratio = ((op.w_in - 1) as f32) / ((op.w_out - 1) as f32);

        for b in 0..op.batch {
            for c in 0..op.chan {
                let i_base = b * istr[0] + c * istr[1];
                for y_out in 0..op.h_out {
                    for x_out in 0..op.w_out {
                        let go =
                            grad_out[b * ostr[0] + c * ostr[1] + y_out * ostr[2] + x_out * ostr[3]];

                        let x_frac = x_ratio * x_out as f32;
                        let x0 = x_frac.floor().min((op.w_in - 1) as f32);
                        let x1 = x_frac.ceil().min((op.w_in - 1) as f32);
                        let xw = E::from_f32(x_frac - x0).unwrap();

                        let y_frac = y_ratio * y_out as f32;
                        let y0 = y_frac.floor().min((op.h_in - 1) as f32);
                        let y1 = y_frac.ceil().min((op.h_in - 1) as f32);
                        let yw = E::from_f32(y_frac - y0).unwrap();

                        let [x0, x1, y0, y1] = [x0, x1, y0, y1].map(|q| q as usize);
                        debug_assert!(x0 < op.w_in);
                        debug_assert!(x1 < op.w_in);
                        debug_assert!(y0 < op.h_in);
                        debug_assert!(y1 < op.h_in);

                        grad_inp[i_base + y0 * istr[2] + x0 * istr[3]] +=
                            go * (E::ONE - xw) * (E::ONE - yw);
                        grad_inp[i_base + y0 * istr[2] + x1 * istr[3]] += go * xw * (E::ONE - yw);
                        grad_inp[i_base + y1 * istr[2] + x0 * istr[3]] += go * (E::ONE - xw) * yw;
                        grad_inp[i_base + y1 * istr[2] + x1 * istr[3]] += go * xw * yw;
                    }
                }
            }
        }
        Ok(())
    }
}
