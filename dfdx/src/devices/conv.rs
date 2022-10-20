use super::{AllocateZeros, Cpu};
#[cfg(feature = "cblas")]
use cblas_sys::{
    cblas_sgemm as sgemm, CblasNoTrans as NoTr, CblasRowMajor as RowMajor, CblasTrans as Tr,
};
use std::boxed::Box;

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

impl<const S: usize, const P: usize> DeviceConv2D<S, P> for Cpu
where
    Self: AllocateZeros,
{
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
        let mut patches: Box<
            [[[[[f32; (W + 2 * P - K) / S + 1]; (H + 2 * P - K) / S + 1]; K]; K]; C],
        > = Self::zeros();

        for c in 0..C {
            for k1 in 0..K {
                for k2 in 0..K {
                    for oh in 0..(H + 2 * P - K) / S + 1 {
                        for ow in 0..(W + 2 * P - K) / S + 1 {
                            let y = (oh * S + k1).wrapping_sub(P);
                            let x = (ow * S + k2).wrapping_sub(P);
                            if y < H && x < W {
                                patches[c][k1][k2][oh][ow] = img[c][y][x];
                            }
                        }
                    }
                }
            }
        }

        // (O, C * K * K) * (C * K * K, OH * OW) = (O, OH * OW)
        let m = O;
        let k = C * K * K;
        let n = ((H + 2 * P - K) / S + 1) * ((W + 2 * P - K) / S + 1);
        let a = weight.as_ptr() as *const f32;
        let b = patches.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        #[cfg(not(feature = "cblas"))]
        unsafe {
            matrixmultiply::sgemm(
                m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 1.0, c, n as isize, 1,
            )
        }

        #[cfg(feature = "cblas")]
        unsafe {
            let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
            sgemm(RowMajor, NoTr, NoTr, m, n, k, 1.0, a, k, b, n, 1.0, c, n)
        }

        for oc in 0..O {
            for oh in 0..((H + 2 * P - K) / S + 1) {
                for ow in 0..((W + 2 * P - K) / S + 1) {
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
        for oc in 0..O {
            for oh in 0..((H + 2 * P - K) / S + 1) {
                for ow in 0..((W + 2 * P - K) / S + 1) {
                    bias_g[oc] += out_g[oc][oh][ow];
                }
            }
        }

        let mut w_tr: Box<[[[[f32; K]; K]; O]; C]> = Self::zeros();
        for c in 0..C {
            for o in 0..O {
                w_tr[c][o].clone_from(&weight[o][c]);
            }
        }

        let mut patches: Box<[[[[[f32; W]; H]; K]; K]; O]> = Self::zeros();
        for o in 0..O {
            for oh in 0..(H + 2 * P - K) / S + 1 {
                for ow in 0..(W + 2 * P - K) / S + 1 {
                    let g = out_g[o][oh][ow];
                    for k1 in 0..K {
                        for k2 in 0..K {
                            let y = (oh * S + k1).wrapping_sub(P);
                            let x = (ow * S + k2).wrapping_sub(P);
                            if y < H && x < W {
                                patches[o][k1][k2][y][x] = g;
                            }
                        }
                    }
                }
            }
        }

        {
            // img_g += weight^T * patches
            // (C, H * W) += (C, O * K * K) * (O * K * K, H * W)

            let m = C;
            let k = O * K * K;
            let n = H * W;
            let a = w_tr.as_ptr() as *const f32;
            let b = patches.as_ptr() as *const f32;
            let c = img_g.as_mut_ptr() as *mut f32;
            #[cfg(not(feature = "cblas"))]
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 1.0, c, n as isize, 1,
                )
            }

            #[cfg(feature = "cblas")]
            unsafe {
                let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
                sgemm(RowMajor, NoTr, NoTr, m, n, k, 1.0, a, k, b, n, 1.0, c, n)
            }
        }

        {
            // weight_g^T += img * patches^T
            // (C, O * K * K) += (C, H * W) * (H * W, O * K * K)

            let m = C;
            let k = H * W;
            let n = O * K * K;
            let a = img.as_ptr() as *const f32;
            let b = patches.as_ptr() as *const f32;
            let c = w_tr.as_mut_ptr() as *mut f32;
            #[cfg(not(feature = "cblas"))]
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n, 1.0, a, k as isize, 1, b, 1, k as isize, 0.0, c, n as isize, 1,
                )
            }
            #[cfg(feature = "cblas")]
            unsafe {
                let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
                sgemm(RowMajor, NoTr, Tr, m, n, k, 1.0, a, k, b, k, 0.0, c, n)
            }

            for o in 0..O {
                for c in 0..C {
                    for k1 in 0..K {
                        for k2 in 0..K {
                            weight_g[o][c][k1][k2] += w_tr[c][o][k1][k2];
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
