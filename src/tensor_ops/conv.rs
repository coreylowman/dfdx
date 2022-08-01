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
                            let y = oh * S + k1;
                            let x = ow * S + k2;
                            if P <= y && y < H + P && P <= x && x < W + P {
                                let y = y - P;
                                let x = x - P;
                                out[oc][oh][ow] += weight[oc][c][k1][k2] * img[c][y][x];
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
                            let y = oh * S + k1;
                            let x = ow * S + k2;
                            if P <= y && y < H + P && P <= x && x < W + P {
                                let y = y - P;
                                let x = x - P;
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
                            let y = oh * S + k1;
                            let x = ow * S + k2;
                            if P <= y && y < H + P && P <= x && x < W + P {
                                let y = y - P;
                                let x = x - P;
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
        let weight = Tensor4D::new([
            [[
                [-0.26322937, 0.01512846, 0.09962472],
                [0.10413083, -0.22805870, 0.09276673],
                [-0.21896772, -0.03946638, -0.06927481],
            ]],
            [[
                [0.11536542, -0.18772237, -0.12918231],
                [-0.14172284, 0.01666686, 0.07898781],
                [0.09430301, 0.21543005, 0.11455652],
            ]],
        ]);
        let bias = Tensor1D::new([-0.14851105, -0.16408388]);
        let x = Tensor3D::new([[
            [
                -0.51514906,
                -0.19863479,
                -0.54003549,
                1.28277719,
                -0.01594487,
            ],
            [
                -1.19097006,
                -0.80577898,
                1.43026400,
                -1.21097565,
                1.09168971,
            ],
            [0.46793947, 0.31626424, -0.98594153, -0.57979399, 0.68837667],
        ]]);
        let result = conv2d::<OwnedTape, 1, 2, 3, 1, 0, 3, 5>(x.trace(), &weight, &bias);
        assert_eq!(
            result.data(),
            &[
                [[0.076069996, -0.48920196, 0.7289252]],
                [[0.15118313, -0.4579478, -0.74080986]]
            ]
        );
        let gradients = result.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[[
                [
                    -0.024973392,
                    -0.04840806,
                    -0.10714647,
                    -0.013128357,
                    0.024153749
                ],
                [
                    -0.008748706,
                    -0.042083982,
                    0.03516327,
                    -0.05965724,
                    0.038324557
                ],
                [
                    -0.021096727,
                    0.022234388,
                    -0.039724708,
                    0.008481046,
                    -0.014830692
                ]
            ]]
        );

        assert_eq!(
            gradients.ref_gradient(&weight),
            &[
                [[
                    [-0.29951084, 0.35226136, 0.028453378],
                    [0.19759789, -0.4171204, 0.51062536],
                    [-0.22414659, -0.24417697, 0.0012589097]
                ]],
                [[
                    [-0.16372146, 0.0064775944, 0.029280666],
                    [-0.20220357, -0.10163972, 0.23634934],
                    [0.045724787, -0.088701606, -0.19757581]
                ]]
            ]
        );

        assert_eq!(gradients.ref_gradient(&bias), &[0.62750089, 0.37875301]);
    }

    #[test]
    fn test_conv2d_stride_2() {
        let weight = Tensor4D::new([
            [[
                [0.17012766, -0.13491425, -0.15978174],
                [-0.32486033, -0.04559305, -0.23446295],
                [-0.19051278, 0.26051202, -0.25324664],
            ]],
            [[
                [0.01786593, -0.29400513, -0.05565581],
                [0.19891194, -0.22905394, 0.06598040],
                [0.20475981, -0.27209029, -0.05456376],
            ]],
        ]);
        let bias = Tensor1D::new([0.13324055, -0.25587624]);
        let x = Tensor3D::new([[
            [1.38410544, -0.09391765, 0.86929655, 0.51864260, -0.13863549],
            [
                -0.97774553,
                -0.86014092,
                -0.87199527,
                -0.99487656,
                -0.15869008,
            ],
            [-0.72478843, 0.00621646, 0.34301779, 1.59643257, 0.42176643],
        ]]);
        let result = conv2d::<OwnedTape, 1, 2, 3, 2, 0, 3, 5>(x.trace(), &weight, &bias);
        assert_eq!(
            result.data(),
            &[[[0.8566188, 0.84288383]], [[-0.47573355, -0.7283041]]]
        );
        let gradients = result.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[[
                [
                    0.10294608,
                    -0.1251128,
                    -0.0017652321,
                    -0.11383441,
                    -0.099512145
                ],
                [
                    -0.16037403,
                    -0.062430196,
                    -0.29246253,
                    -0.05412144,
                    -0.12820505
                ],
                [-0.08036223, 0.11111723, -0.2435197, 0.11845972, -0.15366143]
            ]]
        );

        assert_eq!(
            gradients.ref_gradient(&weight),
            &[
                [[
                    [1.319812, 0.2459107, 0.4313238],
                    [-1.0821161, -1.0842361, -0.60558885],
                    [-0.22754006, 0.93081105, 0.44691432]
                ]],
                [[
                    [0.31993905, 0.0479999, 0.11832076],
                    [-0.25713378, -0.25369257, -0.15462178],
                    [-0.071205154, 0.19362603, 0.10418981]
                ]]
            ]
        );

        assert_eq!(gradients.ref_gradient(&bias), &[1.16956019, 0.27603900]);
    }

    #[test]
    fn test_conv2d_padding_1() {
        let weight = Tensor4D::new([
            [
                [[0.10215953, 0.06263646], [-0.04124039, -0.09729567]],
                [[-0.32656857, 0.24254093], [-0.27209827, 0.15361503]],
            ],
            [
                [[0.03449896, 0.22931078], [-0.17652659, 0.08222872]],
                [[-0.06016779, 0.29082409], [-0.19154115, 0.13483226]],
            ],
            [
                [[-0.14262493, 0.19654515], [0.15921101, 0.01759464]],
                [[0.16749159, 0.33096817], [0.28376505, -0.05524009]],
            ],
        ]);
        let bias = Tensor1D::new([-0.22854491, 0.28763595, 0.20709404]);
        let x = Tensor3D::new([[[-0.32224107, -0.32800716]], [[-1.13570976, 0.93713200]]]);
        let result = conv2d::<OwnedTape, 2, 3, 2, 1, 1, 1, 2>(x.trace(), &weight, &bias);
        assert_eq!(
            result.data(),
            &[
                [
                    [-0.37165433, 0.26964033, -0.47000977],
                    [-0.52418506, 0.3161699, -0.56809187]
                ],
                [
                    [0.10800815, 0.66143924, 0.16603859],
                    [-0.11654915, 0.5421771, 0.21993488]
                ],
                [
                    [0.26416105, -0.22402346, 0.420797],
                    [-0.23212466, 0.3085245, 0.41083777]
                ]
            ]
        );
        let gradients = result.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&x),
            &[[[0.010052743, 0.038219165]], [[0.0013861917, 0.096129306]]]
        );

        assert_eq!(
            gradients.ref_gradient(&weight),
            &[
                [
                    [[-0.03488452, -0.035597768], [-0.03483199, -0.036207683]],
                    [[-0.05705857, 0.03406856], [-0.05008337, 0.024666183]]
                ],
                [
                    [[-0.053492695, -0.04727108], [-0.05620105, -0.055251926]],
                    [[-0.04363727, 0.033381317], [-0.0607851, 0.030584559]]
                ],
                [
                    [[-0.051853612, -0.03900232], [-0.04206547, -0.037880093]],
                    [[-0.0073834136, 0.0208545], [0.02886929, -0.040557314]]
                ]
            ]
        );

        assert_eq!(
            gradients.ref_gradient(&bias),
            &[0.28636602, 0.44933242, 0.40484178]
        );
    }

    #[test]
    fn test_conv2d_stride_3_padding_3() {
        todo!();
    }

    #[test]
    fn test_batched_conv2d() {
        todo!();
    }
}
