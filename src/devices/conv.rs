use super::Cpu;

/// **Requires nightly** 2d convolution with stride and padding specified at trait level.
///
/// This allows the rest of the parameters to be inferred by inputs.
pub trait DeviceConv2D<const S: usize, const P: usize> {
    /// Forward operation that modifies the `out` image.
    fn conv_forward<
        const C: usize,
        const O: usize,
        const K: usize,
        const H: usize,
        const W: usize,
    >(
        img: &[[[f32; W]; H]; C],
        weight: &[[[[f32; K]; K]; C]; O],
        bias: &[f32; O],
        out: &mut [[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; O],
    );

    /// Backward operation that modifies the gradients of img, weight, and bias.
    fn conv_backward<
        const C: usize,
        const O: usize,
        const K: usize,
        const H: usize,
        const W: usize,
    >(
        img: &[[[f32; W]; H]; C],
        weight: &[[[[f32; K]; K]; C]; O],
        out_g: &[[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; O],
        img_g: &mut [[[f32; W]; H]; C],
        weight_g: &mut [[[[f32; K]; K]; C]; O],
        bias_g: &mut [f32; O],
    );
}

impl<const S: usize, const P: usize> DeviceConv2D<S, P> for Cpu {
    fn conv_forward<
        const C: usize,
        const O: usize,
        const K: usize,
        const H: usize,
        const W: usize,
    >(
        img: &[[[f32; W]; H]; C],
        weight: &[[[[f32; K]; K]; C]; O],
        bias: &[f32; O],
        out: &mut [[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; O],
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oc in 0..O {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let o = &mut out[oc][oh][ow];
                        for k1 in 0..K {
                            let y = (oh * S + k1).checked_sub(P);
                            for k2 in 0..K {
                                let x = (ow * S + k2).checked_sub(P);
                                if let Some((y, x)) = y.zip(x) {
                                    if y < H && x < W {
                                        *o += weight[oc][c][k1][k2] * img[c][y][x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for oc in 0..O {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    out[oc][oh][ow] += bias[oc];
                }
            }
        }
    }

    fn conv_backward<
        const C: usize,
        const O: usize,
        const K: usize,
        const H: usize,
        const W: usize,
    >(
        img: &[[[f32; W]; H]; C],
        weight: &[[[[f32; K]; K]; C]; O],
        out_g: &[[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; O],
        img_g: &mut [[[f32; W]; H]; C],
        weight_g: &mut [[[[f32; K]; K]; C]; O],
        bias_g: &mut [f32; O],
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;

        for oc in 0..O {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    bias_g[oc] += out_g[oc][oh][ow];
                }
            }
        }

        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    for oc in 0..O {
                        let o_g = &out_g[oc][oh][ow];
                        for k1 in 0..K {
                            let y = (oh * S + k1).wrapping_sub(P);
                            if y < H {
                                for k2 in 0..K {
                                    let x = (ow * S + k2).wrapping_sub(P);
                                    if x < W {
                                        weight_g[oc][c][k1][k2] += img[c][y][x] * o_g;
                                        img_g[c][y][x] += weight[oc][c][k1][k2] * o_g;
                                    }
                                }
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
    use crate::devices::{AllocateZeros, FillElements};
    use crate::tests::assert_close;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    #[test]
    fn test_conv2d_s4p3k2() {
        let mut rng = StdRng::seed_from_u64(432);
        let mut randn = |x: &mut f32| *x = rng.sample(StandardNormal);

        let weight: Box<[[[[f32; 2]; 2]; 5]; 3]> = Cpu::filled(&mut randn);
        let bias: Box<[f32; 3]> = Cpu::filled(&mut randn);
        let x: Box<[[[f32; 6]; 7]; 5]> = Cpu::filled(&mut randn);

        let mut out = [[[0.0; 3]; 3]; 3];
        <Cpu as DeviceConv2D<4, 3>>::conv_forward(
            x.as_ref(),
            weight.as_ref(),
            bias.as_ref(),
            &mut out,
        );

        #[rustfmt::skip]
        assert_close(&out, &[
            [[-0.57176435, -0.57176435, -0.57176435],[-0.57176435, 1.0759051, 1.4307989],[-0.57176435, -0.86296344, -1.8794353]],
            [[0.29306656, 0.29306656, 0.29306656],[0.29306656, 0.9771965, 1.467767],[0.29306656, -6.367015, -2.3370528]],
            [[-0.19717735, -0.19717735, -0.19717735],[-0.19717735, 1.3412137, 2.9476144],[-0.19717735, 4.247249, -2.1779637]],
        ]);

        let mut wg: Box<[[[[f32; 2]; 2]; 5]; 3]> = Cpu::zeros();
        let mut bg: Box<[f32; 3]> = Cpu::zeros();
        let mut xg: Box<[[[f32; 6]; 7]; 5]> = Cpu::zeros();
        <Cpu as DeviceConv2D<4, 3>>::conv_backward(
            &x,
            &weight,
            &out,
            xg.as_mut(),
            wg.as_mut(),
            bg.as_mut(),
        );

        assert_ne!(wg.as_ref(), &[[[[0.0; 2]; 2]; 5]; 3]);
        assert_ne!(bg.as_ref(), &[0.0; 3]);
        assert_ne!(xg.as_ref(), &[[[0.0; 6]; 7]; 5]);
    }
}
