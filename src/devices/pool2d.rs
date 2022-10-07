use super::Cpu;

pub struct PoolMax;
pub struct PoolMin;
pub struct PoolAvg;

/// **Requires nightly** 2d convolution with stride and padding specified at trait level.
///
/// This allows the rest of the parameters to be inferred by inputs.
pub trait DevicePool2D<const K: usize, const S: usize, const P: usize, Kind> {
    /// Forward operation that modifies the `out` image.
    fn pool_forward<const C: usize, const H: usize, const W: usize>(
        inp: &[[[f32; W]; H]; C],
        out: &mut [[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; C],
    );

    /// Backward operation that modifies the gradients of img, weight, and bias.
    fn pool_backward<const C: usize, const H: usize, const W: usize>(
        inp: &[[[f32; W]; H]; C],
        out_g: &[[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; C],
        inp_g: &mut [[[f32; W]; H]; C],
    );
}

impl<const K: usize, const S: usize, const P: usize> DevicePool2D<K, S, P, PoolMax> for Cpu {
    fn pool_forward<const C: usize, const H: usize, const W: usize>(
        inp: &[[[f32; W]; H]; C],
        out: &mut [[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; C],
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let o = &mut out[c][oh][ow];
                    let mut tmp = f32::NEG_INFINITY;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp = tmp.max(inp[c][y][x]);
                                }
                            }
                        }
                    }
                    *o = tmp;
                }
            }
        }
    }

    fn pool_backward<const C: usize, const H: usize, const W: usize>(
        inp: &[[[f32; W]; H]; C],
        out_g: &[[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; C],
        inp_g: &mut [[[f32; W]; H]; C],
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let o_g = &out_g[c][oh][ow];
                    let mut tmp = f32::NEG_INFINITY;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp = tmp.max(inp[c][y][x]);
                                }
                            }
                        }
                    }

                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W && inp[c][y][x] == tmp {
                                    inp_g[c][y][x] += o_g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<const K: usize, const S: usize, const P: usize> DevicePool2D<K, S, P, PoolMin> for Cpu {
    fn pool_forward<const C: usize, const H: usize, const W: usize>(
        inp: &[[[f32; W]; H]; C],
        out: &mut [[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; C],
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let o = &mut out[c][oh][ow];
                    let mut tmp = f32::INFINITY;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp = tmp.min(inp[c][y][x]);
                                }
                            }
                        }
                    }
                    *o = tmp;
                }
            }
        }
    }

    fn pool_backward<const C: usize, const H: usize, const W: usize>(
        inp: &[[[f32; W]; H]; C],
        out_g: &[[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; C],
        inp_g: &mut [[[f32; W]; H]; C],
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let o_g = &out_g[c][oh][ow];
                    let mut tmp = f32::INFINITY;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp = tmp.min(inp[c][y][x]);
                                }
                            }
                        }
                    }

                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W && inp[c][y][x] == tmp {
                                    inp_g[c][y][x] += o_g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<const K: usize, const S: usize, const P: usize> DevicePool2D<K, S, P, PoolAvg> for Cpu {
    fn pool_forward<const C: usize, const H: usize, const W: usize>(
        inp: &[[[f32; W]; H]; C],
        out: &mut [[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; C],
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        let inv_k2 = 1.0 / (K * K) as f32;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let o = &mut out[c][oh][ow];
                    let mut tmp = 0.0;
                    for k1 in 0..K {
                        let y = (oh * S + k1).checked_sub(P);
                        for k2 in 0..K {
                            let x = (ow * S + k2).checked_sub(P);
                            if let Some((y, x)) = y.zip(x) {
                                if y < H && x < W {
                                    tmp += inp[c][y][x];
                                }
                            }
                        }
                    }
                    *o = tmp * inv_k2;
                }
            }
        }
    }

    fn pool_backward<const C: usize, const H: usize, const W: usize>(
        _inp: &[[[f32; W]; H]; C],
        out_g: &[[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; C],
        inp_g: &mut [[[f32; W]; H]; C],
    ) {
        let out_height = (H + 2 * P - K) / S + 1;
        let out_width = (W + 2 * P - K) / S + 1;
        let inv_k2 = 1.0 / (K * K) as f32;
        for c in 0..C {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let g = out_g[c][oh][ow] * inv_k2;
                    for k1 in 0..K {
                        let y = (oh * S + k1).wrapping_sub(P);
                        if y < H {
                            for k2 in 0..K {
                                let x = (ow * S + k2).wrapping_sub(P);
                                if x < W {
                                    inp_g[c][y][x] += g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
