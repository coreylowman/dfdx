use crate::shapes::*;
use crate::tensor::cpu::Cpu;

use std::sync::Arc;

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => panic!("Only implemented for 3d & 4d arrays"),
    }
}

impl super::AvgPool2DKernel<f32> for Cpu {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let mut tmp = 0.0;
                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if y < op.h_in && x < op.w_in {
                                        let inp_idx =
                                            b * istr[0] + c * istr[1] + y * istr[2] + x * istr[3];
                                        tmp += buf[inp_idx];
                                    }
                                }
                            }
                        }
                        tmp /= (op.kernel * op.kernel) as f32;
                        out_buf[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]] = tmp;
                    }
                }
            }
        }
        Ok(())
    }

    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        grad_inp: &mut Self::Storage<I, f32>,
        out: &Self::Storage<O, f32>,
        grad_out: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let ginp_buf = Arc::make_mut(&mut grad_inp.data);
        let buf = grad_out.data.as_ref();

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let g = buf[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]]
                            / (op.kernel * op.kernel) as f32;

                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if x < op.w_in && y < op.h_in {
                                        ginp_buf[b * istr[0]
                                            + c * istr[1]
                                            + y * istr[2]
                                            + x * istr[3]] += g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl super::MaxPool2DKernel<f32> for Cpu {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let mut tmp = f32::NEG_INFINITY;
                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if y < op.h_in && x < op.w_in {
                                        tmp = tmp.max(
                                            buf[b * istr[0]
                                                + c * istr[1]
                                                + y * istr[2]
                                                + x * istr[3]],
                                        );
                                    }
                                }
                            }
                        }
                        out_buf[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]] = tmp;
                    }
                }
            }
        }
        Ok(())
    }
    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        grad_inp: &mut Self::Storage<I, f32>,
        out: &Self::Storage<O, f32>,
        grad_out: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let inp_buf = inp.data.as_ref();
        let ginp_buf = Arc::make_mut(&mut grad_inp.data);
        let out_buf = out.data.as_ref();
        let gout_buf = grad_out.data.as_ref();

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let out_idx = b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3];
                        let go = gout_buf[out_idx];
                        let vo = out_buf[out_idx];
                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if x < op.w_in && y < op.h_in {
                                        let inp_idx =
                                            b * istr[0] + c * istr[1] + y * istr[2] + x * istr[3];
                                        if inp_buf[inp_idx] == vo {
                                            ginp_buf[inp_idx] += go;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl super::MinPool2DKernel<f32> for Cpu {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        out: &mut Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let mut tmp = f32::INFINITY;
                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if y < op.h_in && x < op.w_in {
                                        tmp = tmp.min(
                                            buf[b * istr[0]
                                                + c * istr[1]
                                                + y * istr[2]
                                                + x * istr[3]],
                                        );
                                    }
                                }
                            }
                        }
                        out_buf[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]] = tmp;
                    }
                }
            }
        }
        Ok(())
    }
    fn backward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Self::Storage<I, f32>,
        grad_inp: &mut Self::Storage<I, f32>,
        out: &Self::Storage<O, f32>,
        grad_out: &Self::Storage<O, f32>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let inp_buf = inp.data.as_ref();
        let ginp_buf = Arc::make_mut(&mut grad_inp.data);
        let out_buf = out.data.as_ref();
        let gout_buf = grad_out.data.as_ref();

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let out_idx = b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3];
                        let go = gout_buf[out_idx];
                        let vo = out_buf[out_idx];
                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if x < op.w_in && y < op.h_in {
                                        let inp_idx =
                                            b * istr[0] + c * istr[1] + y * istr[2] + x * istr[3];
                                        if inp_buf[inp_idx] == vo {
                                            ginp_buf[inp_idx] += go;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
