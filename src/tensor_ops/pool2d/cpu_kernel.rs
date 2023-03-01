use crate::shapes::*;
use crate::tensor::{Cpu, Tensor};

use std::sync::Arc;

use num_traits::Float;

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => panic!("Only implemented for 3d & 4d arrays"),
    }
}

impl<E: Float + Unit + std::ops::AddAssign + std::ops::DivAssign> super::AvgPool2DKernel<E>
    for Cpu
{
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let mut tmp = E::zero();
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
                        tmp /= E::from(op.kernel * op.kernel).unwrap();
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
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let g = grad_out[b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3]]
                            / E::from(op.kernel * op.kernel).unwrap();

                        for k1 in 0..op.kernel {
                            let y = (oh * op.stride + k1).checked_sub(op.padding);
                            for k2 in 0..op.kernel {
                                let x = (ow * op.stride + k2).checked_sub(op.padding);
                                if let Some((y, x)) = y.zip(x) {
                                    if x < op.w_in && y < op.h_in {
                                        grad_inp[b * istr[0]
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

impl<E: Float + Unit + std::ops::AddAssign> super::MaxPool2DKernel<E> for Cpu {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let mut tmp = E::neg_infinity();
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
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let inp_buf = inp.data.as_ref();
        let out_buf = out.data.as_ref();

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let out_idx = b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3];
                        let go = grad_out[out_idx];
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
                                            grad_inp[inp_idx] += go;
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

impl<E: Float + Unit + std::ops::AddAssign> super::MinPool2DKernel<E> for Cpu {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: super::Pool2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let buf = inp.data.as_ref();
        let out_buf = Arc::make_mut(&mut out.data);
        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let mut tmp = E::infinity();
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
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let istr = make_4d::<I>(inp.strides);
        let ostr = make_4d::<O>(out.strides);

        let inp_buf = inp.data.as_ref();
        let out_buf = out.data.as_ref();

        for b in 0..op.batch {
            for c in 0..op.chan {
                for oh in 0..op.h_out {
                    for ow in 0..op.w_out {
                        let out_idx = b * ostr[0] + c * ostr[1] + oh * ostr[2] + ow * ostr[3];
                        let go = grad_out[out_idx];
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
                                            grad_inp[inp_idx] += go;
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
