use crate::prelude::*;

pub fn conv2d<
    TAPE: 'static + Tape,
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL: usize,
    const STRIDE: usize,
    const PADDING: usize,
    const IN_HEIGHT: usize,
    const IN_WIDTH: usize,
>(
    x: Tensor3D<IN_CHAN, IN_HEIGHT, IN_WIDTH, TAPE>,
    filters: &Tensor4D<OUT_CHAN, IN_CHAN, KERNEL, KERNEL>,
    bias: &Tensor1D<OUT_CHAN>,
) -> Tensor3D<
    OUT_CHAN,
    { (IN_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1 },
    { (IN_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1 },
    TAPE,
> {
    let mut result = Tensor3D::zeros();
    conv_forward::<
        IN_CHAN,
        OUT_CHAN,
        KERNEL,
        STRIDE,
        PADDING,
        IN_HEIGHT,
        IN_WIDTH,
        { (IN_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1 },
        { (IN_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1 },
    >(x.data(), filters.data(), bias.data(), result.mut_data());

    let filters_data = filters.data.clone();

    let (x, mut tape) = x.split_tape();
    let phantom_filters = filters.phantom();
    let phantom_bias = bias.phantom();
    let phantom_result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (filters_grad, result_grad) = grads.mut_and_ref(&phantom_filters, &phantom_result);
        conv_backward_dw::<
            IN_CHAN,
            OUT_CHAN,
            KERNEL,
            STRIDE,
            PADDING,
            IN_HEIGHT,
            IN_WIDTH,
            { (IN_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1 },
            { (IN_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1 },
        >(x.data(), filters_grad, result_grad);

        let (bias_grad, result_grad) = grads.mut_and_ref(&phantom_bias, &phantom_result);
        conv_backward_db(bias_grad, result_grad);

        let (inp_grad, result_grad) = grads.mut_and_ref(&x, &phantom_result);
        conv_backward_dx::<
            IN_CHAN,
            OUT_CHAN,
            KERNEL,
            STRIDE,
            PADDING,
            IN_HEIGHT,
            IN_WIDTH,
            { (IN_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1 },
            { (IN_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1 },
        >(inp_grad, filters_data.as_ref(), result_grad);
    });
    result.put_tape(tape)
}

pub fn batch_conv2d<
    TAPE: 'static + Tape,
    const BATCH_SIZE: usize,
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL: usize,
    const STRIDE: usize,
    const PADDING: usize,
    const IN_HEIGHT: usize,
    const IN_WIDTH: usize,
>(
    x: Tensor4D<BATCH_SIZE, IN_CHAN, IN_HEIGHT, IN_WIDTH, TAPE>,
    filters: &Tensor4D<OUT_CHAN, IN_CHAN, KERNEL, KERNEL>,
    bias: &Tensor1D<OUT_CHAN>,
) -> Tensor4D<
    BATCH_SIZE,
    OUT_CHAN,
    { (IN_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1 },
    { (IN_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1 },
    TAPE,
> {
    let mut result = Tensor4D::zeros();
    for i in 0..BATCH_SIZE {
        conv_forward::<
            IN_CHAN,
            OUT_CHAN,
            KERNEL,
            STRIDE,
            PADDING,
            IN_HEIGHT,
            IN_WIDTH,
            { (IN_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1 },
            { (IN_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1 },
        >(
            &x.data()[i],
            filters.data(),
            bias.data(),
            &mut result.mut_data()[i],
        );
    }

    let filters_data = filters.data.clone();

    let (x, mut tape) = x.split_tape();
    let phantom_filters = filters.phantom();
    let phantom_bias = bias.phantom();
    let phantom_result = result.phantom();
    tape.add_backward_op(move |grads| {
        let (filters_grad, result_grad) = grads.mut_and_ref(&phantom_filters, &phantom_result);
        for i in 0..BATCH_SIZE {
            conv_backward_dw::<
                IN_CHAN,
                OUT_CHAN,
                KERNEL,
                STRIDE,
                PADDING,
                IN_HEIGHT,
                IN_WIDTH,
                { (IN_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1 },
                { (IN_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1 },
            >(&x.data()[i], filters_grad, &result_grad[i]);
        }

        let (bias_grad, result_grad) = grads.mut_and_ref(&phantom_bias, &phantom_result);
        for i in 0..BATCH_SIZE {
            conv_backward_db(bias_grad, &result_grad[i]);
        }

        let (inp_grad, result_grad) = grads.mut_and_ref(&x, &phantom_result);
        for i in 0..BATCH_SIZE {
            conv_backward_dx::<
                IN_CHAN,
                OUT_CHAN,
                KERNEL,
                STRIDE,
                PADDING,
                IN_HEIGHT,
                IN_WIDTH,
                { (IN_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1 },
                { (IN_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1 },
            >(&mut inp_grad[i], filters_data.as_ref(), &result_grad[i]);
        }
    });
    result.put_tape(tape)
}

fn conv_forward<
    const C: usize,
    const OC: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
    const OH: usize,
    const OW: usize,
>(
    img: &[[[f32; W]; H]; C],
    weight: &[[[[f32; K]; K]; C]; OC],
    bias: &[f32; OC],
    out: &mut [[[f32; OW]; OH]; OC],
) {
    for oc in 0..OC {
        for oh in 0..OH {
            for ow in 0..OW {
                out[oc][oh][ow] += bias[oc];
                for c in 0..C {
                    for k1 in 0..K {
                        for k2 in 0..K {
                            let y = (oh * S + k1).checked_sub(2 * P);
                            let x = (ow * S + k2).checked_sub(2 * P);
                            if let Some((y, x)) = y.zip(x) {
                                let w = weight[oc][c][k1][k2];
                                let v = img[c][y][x];
                                out[oc][oh][ow] += w * v;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn conv_backward_dw<
    const C: usize,
    const OC: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
    const OH: usize,
    const OW: usize,
>(
    img: &[[[f32; W]; H]; C],
    weight: &mut [[[[f32; K]; K]; C]; OC],
    out: &[[[f32; OW]; OH]; OC],
) {
    for oc in 0..OC {
        for oh in 0..OH {
            for ow in 0..OW {
                for c in 0..C {
                    for k1 in 0..K {
                        for k2 in 0..K {
                            let y = (oh * S + k1).checked_sub(2 * P);
                            let x = (ow * S + k2).checked_sub(2 * P);
                            if let Some((y, x)) = y.zip(x) {
                                weight[oc][c][k1][k2] += img[c][y][x] * out[oc][oh][ow];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn conv_backward_db<const OC: usize, const OH: usize, const OW: usize>(
    bias: &mut [f32; OC],
    out: &[[[f32; OW]; OH]; OC],
) {
    for oc in 0..OC {
        for oh in 0..OH {
            for ow in 0..OW {
                bias[oc] += out[oc][oh][ow];
            }
        }
    }
}

fn conv_backward_dx<
    const C: usize,
    const OC: usize,
    const K: usize,
    const S: usize,
    const P: usize,
    const H: usize,
    const W: usize,
    const OH: usize,
    const OW: usize,
>(
    img: &mut [[[f32; W]; H]; C],
    weight: &[[[[f32; K]; K]; C]; OC],
    out: &[[[f32; OW]; OH]; OC],
) {
    for oc in 0..OC {
        for oh in 0..OH {
            for ow in 0..OW {
                for c in 0..C {
                    for k1 in 0..K {
                        for k2 in 0..K {
                            let y = (oh * S + k1).checked_sub(2 * P);
                            let x = (ow * S + k2).checked_sub(2 * P);
                            if let Some((y, x)) = y.zip(x) {
                                img[c][y][x] += weight[oc][c][k1][k2] * out[oc][oh][ow];
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_default_stride_and_padding() {
        todo!("");
    }

    #[test]
    fn test_conv2d_stride_2() {
        todo!();
    }

    #[test]
    fn test_conv2d_padding_1() {
        todo!();
    }

    #[test]
    fn test_conv2d_stride_3_padding_3() {
        todo!();
    }
}
