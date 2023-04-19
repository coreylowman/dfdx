mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(super) struct ConvTrans2DOp {
    pub stride: usize,
    pub padding: usize,
    pub kernel: usize,
    pub batch: usize,
    pub chan_in: usize,
    pub chan_out: usize,
    pub h_in: usize,
    pub h_out: usize,
    pub w_in: usize,
    pub w_out: usize,
}

impl ConvTrans2DOp {
    fn new(s: usize, p: usize, k: usize, [b, c, h_in, w_in]: [usize; 4], o: usize) -> Self {
        Self {
            stride: s,
            padding: p,
            kernel: k,
            batch: b,
            chan_in: c,
            chan_out: o,
            h_in,
            h_out: (h_in - 1) * s - 2 * p + k,
            w_in,
            w_out: (w_in - 1) * s - 2 * p + k,
        }
    }

    #[rustfmt::skip]
    pub(super) fn inp_patches_shape(&self) -> (usize, usize, usize, usize, usize) {
        (self.chan_in, self.kernel, self.kernel, self.h_out, self.w_out)
    }

    #[rustfmt::skip]
    pub(super) fn out_patches_shape(&self) -> (usize, usize, usize, usize, usize) {
        (self.chan_out, self.kernel, self.kernel, self.h_in, self.w_in)
    }

    pub(super) fn filters_tr_shape(&self) -> (usize, usize, usize, usize) {
        (self.chan_in, self.chan_out, self.kernel, self.kernel)
    }
}

pub(super) trait ConvTrans2DKernel<E: Dtype>: DeviceStorage {
    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err>;

    #[allow(clippy::too_many_arguments)]
    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

pub trait ConvTransAlgebra<const K: usize, const S: usize, const P: usize>: Dim {
    type Convolved: Dim;

    fn convolve_dim(&self) -> Self::Convolved;
}

impl<const D: usize, const K: usize, const S: usize, const P: usize> ConvTransAlgebra<K, S, P>
    for Const<D>
where
    Const<{ D * S + K - S - 2 * P }>: Sized,
{
    type Convolved = Const<{ D * S + K - S - 2 * P }>;

    fn convolve_dim(&self) -> Self::Convolved {
        Default::default()
    }
}

impl<const K: usize, const S: usize, const P: usize> ConvTransAlgebra<K, S, P> for usize {
    type Convolved = usize;

    fn convolve_dim(&self) -> Self::Convolved {
        (self * S + K).checked_sub(S + 2 * P).unwrap()
    }
}

pub trait TryConvTrans2DTo<F, const S: usize, const P: usize>: HasErr {
    type Output;
    fn convtrans2d_to(self, filters: F) -> Self::Output {
        self.try_convtrans2d_to(filters).unwrap()
    }
    fn try_convtrans2d_to(self, filters: F) -> Result<Self::Output, Self::Err>;
}

pub trait TryConvTrans2D<F> {
    fn convtrans2d<const S: usize, const P: usize>(self, filters: F) -> Self::Output
    where
        Self: TryConvTrans2DTo<F, S, P>,
    {
        self.convtrans2d_to(filters)
    }
    fn try_convtrans2d<const S: usize, const P: usize>(
        self,
        filters: F,
    ) -> Result<Self::Output, Self::Err>
    where
        Self: TryConvTrans2DTo<F, S, P>,
    {
        self.try_convtrans2d_to(filters)
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, T, F> TryConvTrans2D<F> for Tensor<S, E, D, T> {}

impl<
        const C: usize,
        H: Dim + ConvTransAlgebra<K, S, P>,
        W: Dim + ConvTransAlgebra<K, S, P>,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        E: Dtype,
        D: ConvTrans2DKernel<E> + ZerosTensor<E>,
        T: 'static + Tape<E, D>,
    > TryConvTrans2DTo<Tensor<Rank4<O, C, K, K>, E, D>, S, P>
    for Tensor<(Const<C>, H, W), E, D, T>
{
    type Output = Tensor<(Const<O>, H::Convolved, W::Convolved), E, D, T>;

    fn try_convtrans2d_to(
        self,
        filters: Tensor<Rank4<O, C, K, K>, E, D>,
    ) -> Result<Self::Output, Self::Err> {
        let h = self.shape.1;
        let w = self.shape.2;

        let op = ConvTrans2DOp::new(S, P, K, [1, C, h.size(), w.size()], O);
        let (lhs, ltape) = self.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut tape = ltape.merge(rtape);
        let mut out = lhs
            .device
            .try_zeros_like(&(Const, h.convolve_dim(), w.convolve_dim()))?;
        lhs.device.forward(op, &lhs, &rhs, &mut out)?;
        let lhs_ghost = lhs.ghost();
        let rhs_ghost = rhs.ghost();
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs.device
                .backward(op, &lhs, grad_lhs, &rhs, grad_rhs, &out_ghost, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

impl<
        B: Dim,
        const C: usize,
        H: Dim + ConvTransAlgebra<K, S, P>,
        W: Dim + ConvTransAlgebra<K, S, P>,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        E: Dtype,
        D: ConvTrans2DKernel<E> + ZerosTensor<E>,
        T: 'static + Tape<E, D>,
    > TryConvTrans2DTo<Tensor<Rank4<O, C, K, K>, E, D>, S, P>
    for Tensor<(B, Const<C>, H, W), E, D, T>
{
    type Output = Tensor<(B, Const<O>, H::Convolved, W::Convolved), E, D, T>;
    fn try_convtrans2d_to(
        self,
        filters: Tensor<Rank4<O, C, K, K>, E, D>,
    ) -> Result<Self::Output, Self::Err> {
        let h = self.shape.2;
        let w = self.shape.3;

        let batch = self.shape().0;
        let op = ConvTrans2DOp::new(S, P, K, [batch.size(), C, h.size(), w.size()], O);
        let (lhs, ltape) = self.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut out =
            lhs.device
                .try_zeros_like(&(batch, Const, h.convolve_dim(), w.convolve_dim()))?;
        let mut tape = ltape.merge(rtape);
        lhs.device.forward(op, &lhs, &rhs, &mut out)?;
        let lhs_ghost = lhs.ghost();
        let rhs_ghost = rhs.ghost();
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs.device
                .backward(op, &lhs, grad_lhs, &rhs, grad_rhs, &out_ghost, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    /// TODO
    /// Produced by
    /// ```python
    /// x = torch.rand((2, 3, 3), requires_grad=True)
    /// w = torch.rand((2, 3, 3, 3), requires_grad=True)
    /// print(x)
    /// print(torch.swapaxes(w, 0, 1))
    /// y = torch.conv_transpose2d(x, w)
    /// print(y)
    /// y.exp().mean().backward()
    /// print(x.grad)
    /// print(torch.swapaxes(w.grad, 0, 1))
    /// ```
    fn convtrans2d_test() {
        let device = TestDevice::default();

        #[rustfmt::skip]
        let x = device.tensor([
            [[0.0907329, 0.5784497, 0.1818193], [0.9867508, 0.0566732, 0.9057426], [0.3095418, 0.7836370, 0.3519793]],
            [[0.0969202, 0.6929486, 0.7536632], [0.8163304, 0.4960053, 0.4525285], [0.2100813, 0.2163504, 0.3710884]],
        ]);
        #[rustfmt::skip]
        let w = device.tensor([
            [[[0.3401043, 0.4955706, 0.2441649], [0.4701799, 0.6495003, 0.2529446], [0.4017784, 0.1768293, 0.2353096]], [[0.6621207, 0.7709801, 0.5986264], [0.2237560, 0.6466616, 0.7321741], [0.1578425, 0.9478191, 0.9861543]]],
            [[[0.8291070, 0.0848221, 0.3680936], [0.4642293, 0.1073243, 0.1073309], [0.7863810, 0.3699800, 0.4956312]], [[0.0681600, 0.5616951, 0.4053129], [0.1850831, 0.8223089, 0.0667553], [0.8905262, 0.6328429, 0.8180532]]],
            [[[0.7582999, 0.9763424, 0.5727801], [0.3743349, 0.4793805, 0.6885015], [0.8183323, 0.1882774, 0.9794642]], [[0.5606869, 0.7552301, 0.6572021], [0.8761331, 0.2401637, 0.1778120], [0.2065960, 0.4133974, 0.8821540]]],
        ]);
        let y = x.leaky_trace().convtrans2d::<1, 0>(w.clone());
        #[rustfmt::skip]
        assert_close_to_literal!(
            y,
            [
                [[0.0950315, 0.7752370, 1.4619386, 1.2272182, 0.4955567], [0.9404547, 2.0147018, 2.9196219, 2.3676410, 1.0898490], [0.9427375, 2.4812443, 3.9218845, 3.6057489, 1.6545289], [0.7178541, 1.8030399, 3.0822182, 1.9167527, 1.0201255], [0.1575270, 0.6028528, 0.8236330, 0.8117172, 0.4487746]],
                [[0.0818333, 0.5889638, 0.7130897, 0.9325359, 0.3723960], [0.9338222, 1.1092455, 2.6313026, 1.3005096, 0.5866395], [1.0377907, 2.8708880, 3.3737209, 2.5207422, 1.1140431], [1.6855066, 1.9777625, 3.1483138, 1.4968101, 0.8816571], [0.4305007, 1.0563757, 1.3593760, 0.9304471, 0.4780220]],
                [[0.1231446, 0.9889490, 1.7642095, 1.5334388, 0.5994515], [1.3248383, 2.7914243, 3.7239599, 2.3741565, 1.0753872], [1.5313777, 2.9749527, 4.2994099, 3.4086916, 1.9924896], [1.2760720, 1.3538387, 3.8719988, 1.6865263, 1.5946647], [0.2967100, 0.8310994, 1.0901904, 1.1780756, 0.6721083]],
            ]
        );

        let g = y.exp().mean().backward();

        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[2.4066830, 2.4581399, 2.5645943], [3.0028410, 3.2547507, 3.4216807], [2.1464431, 3.0581608, 2.8662176]],
                [[2.9864695, 3.5932014, 2.5797451], [3.3677268, 3.8909531, 3.2242548], [2.4629762, 3.3527191, 2.9590628]],
            ]
        );
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&w),
            [
                [[[0.6642238, 1.0354118, 0.9408946], [0.9326871, 1.1026800, 1.0413336], [0.5472590, 0.7134937, 0.7234858]], [[0.5456561, 0.7068147, 0.6173539], [0.8016681, 1.0878984, 1.0714644], [0.8350498, 1.1260045, 0.7775879]]],
                [[[0.5567597, 0.5549879, 0.4975571], [0.6702054, 0.8184303, 0.6338357], [0.6227797, 0.5077031, 0.5278049]], [[0.3733614, 0.3889205, 0.3597363], [0.6457027, 0.7389204, 0.5783513], [0.7389930, 0.7089815, 0.5071381]]],
                [[[1.1678052, 1.4273405, 1.2900156], [1.4765850, 1.5869446, 1.4983673], [1.0089380, 0.8733283, 1.0910161]], [[0.9175905, 1.0371233, 0.9381008], [1.4550014, 1.5706275, 1.4026034], [1.3066854, 1.4330946, 1.0638479]]],
            ]
        );
    }

    #[test]
    /// torch.set_printoptions(precision=7)
    /// x = torch.rand((2, 3, 3), requires_grad=True)
    /// w = torch.rand((2, 3, 3, 3), requires_grad=True)
    /// print(x)
    /// print(torch.swapaxes(w, 0, 1))
    /// y = torch.conv_transpose2d(x, w, stride=2)
    /// print(y)
    /// y.exp().mean().backward()
    /// print(x.grad)
    /// print(torch.swapaxes(w.grad, 0, 1))
    fn convtrans2d_s2() {
        let device = TestDevice::default();

        #[rustfmt::skip]
        let x = device.tensor([
            [[0.0357635, 0.0225288, 0.3642959],[0.6850907, 0.2586224, 0.2234361],[0.9315249, 0.7850553, 0.7588840]],
            [[0.1240514, 0.4712945, 0.8732865],[0.1023245, 0.1211519, 0.0407664],[0.5106173, 0.9263544, 0.3101138]],
        ]);
        #[rustfmt::skip]
        let w = device.tensor([
            [[[0.7007528, 0.1896583, 0.9991148],[0.6587640, 0.9383754, 0.3999129],[0.0035173, 0.4376699, 0.3985791],],[[0.2180834, 0.4829719, 0.0272914],[0.2712103, 0.8577049, 0.8002768],[0.7074867, 0.4011419, 0.8835942],],],
            [[[0.0067961, 0.4006048, 0.3549793],[0.5392876, 0.3803764, 0.6090584],[0.4874769, 0.5006863, 0.8963661],],[[0.3751084, 0.5425243, 0.5102475],[0.6024926, 0.2719866, 0.9794098],[0.2236674, 0.1083973, 0.4948432],],],
            [[[0.9486710, 0.9823384, 0.5994584],[0.2740490, 0.2620903, 0.2716798],[0.3620688, 0.9108542, 0.9017550],],[[0.6089512, 0.4252676, 0.2729263],[0.8855131, 0.3937372, 0.3419960],[0.8216078, 0.6664743, 0.5395248],],],
        ]);
        let y = x.leaky_trace().convtrans2d::<2, 0>(w.clone());
        #[rustfmt::skip]
        assert_close_to_literal!(
            y,
            [
                [[0.0521149, 0.0666962, 0.1576860, 0.2318948, 0.4811018, 0.4908646, 0.3878066,],[0.0572037, 0.1399591, 0.2562388, 0.4253721, 0.8630048, 1.0908685, 0.8445575,],[0.5902850, 0.2447678, 1.3523079, 0.3064790, 1.4716963, 0.5718186, 1.1411824,],[0.4790645, 0.7306364, 0.5590932, 0.3465975, 0.3586293, 0.2446325, 0.1219794,],[0.8389287, 0.7641755, 2.1468871, 0.7580858, 1.6488208, 0.4078493, 0.8917535,],[0.7521397, 1.3120791, 1.5495670, 1.5312154, 1.6393251, 0.9781042, 0.5516644,],[0.3645314, 0.6125304, 1.4806095, 0.7151946, 1.3534986, 0.4565403, 0.5764900,],],
                [[0.0467758, 0.0816279, 0.2529318, 0.2647139, 0.5785270, 0.6197178, 0.5749097,],[0.0940269, 0.0473439, 0.4393802, 0.1367552, 1.1979207, 0.3760918, 1.0771828,],[0.0882189, 0.3613173, 0.5524452, 0.2317001, 0.7967559, 0.3886862, 0.8587984,],[0.4311106, 0.2884232, 0.7299427, 0.1313255, 0.4212312, 0.0960777, 0.1760126,],[0.5547202, 1.0043029, 1.7619288, 0.9596879, 1.2826418, 0.5885472, 0.6480764,],[0.8100029, 0.4932112, 2.0489488, 0.5505725, 1.9815230, 0.3730083, 0.7659331,],[0.5683053, 0.5217513, 1.6775545, 0.4934808, 1.6013979, 0.4135783, 0.8336955,],],
                [[0.1094690, 0.0878869, 0.3636634, 0.2225572, 1.0195196, 0.7292423, 0.4567231,],[0.1196501, 0.0582169, 0.4756527, 0.1914707, 1.0404429, 0.4393238, 0.3976323,],[0.8271067, 0.8317585, 1.2522885, 0.6402028, 1.5488806, 1.1506698, 0.9447322,],[0.2783581, 0.2198445, 0.3992766, 0.1154844, 0.2090276, 0.0746117, 0.0746450,],[1.5267723, 1.8244361, 2.8728042, 1.4814504, 2.0451818, 1.1080496, 0.7630367,],[0.7074417, 0.4451926, 1.4631481, 0.5704955, 1.0126743, 0.3209994, 0.3122311,],[0.7568032, 1.1887968, 2.1608419, 1.3324623, 1.7372788, 0.8979155, 0.8516415,],],
            ]
        );

        let g = y.exp().mean().backward();

        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[0.1513395, 0.1986136, 0.2298895],[0.3779295, 0.3469064, 0.2452929],[0.4825282, 0.5639746, 0.3148936],],
                [[0.1527605, 0.2144486, 0.2480491],[0.3177541, 0.3597765, 0.2327209],[0.3772507, 0.5048490, 0.2865718],],
            ]
        );
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&w),
            [
                [[[0.1134962, 0.0483045, 0.1292279],[0.0842974, 0.0839551, 0.0851499],[0.0981565, 0.0517171, 0.1198249],],[[0.0928453, 0.0412302, 0.0897282],[0.0699945, 0.0741750, 0.0777097],[0.0907740, 0.0422520, 0.0901646],],],
                [[[0.0771443, 0.0567608, 0.0866109],[0.1149883, 0.0410739, 0.1213422],[0.0952494, 0.0514846, 0.1152932],],[[0.0687711, 0.0483170, 0.0680406],[0.1007998, 0.0350681, 0.1096105],[0.0770078, 0.0379331, 0.0848609],],],
                [[[0.1948610, 0.1028565, 0.1976164],[0.0683245, 0.0401628, 0.0643963],[0.1662624, 0.1036348, 0.2046718],],[[0.1715550, 0.0769297, 0.1411840],[0.0654903, 0.0355645, 0.0568618],[0.1351082, 0.0760437, 0.1234931],],],
            ]
        );
    }

    #[test]
    fn test_batched_convtrans2d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank3<3, 28, 28>, TestDtype, _> = dev.sample_normal();
        let w: Tensor<Rank4<5, 3, 6, 6>, TestDtype, _> = dev.sample_normal();

        let y: Tensor<Rank3<5, 83, 83>, _, _, _> = x.leaky_trace().convtrans2d::<3, 2>(w.clone());
        let y0 = y.retaped::<NoneTape>();
        let grads0 = y.square().mean().backward();
        let x0 = grads0.get(&x);
        let w0 = grads0.get(&w);

        let x = x
            .broadcast::<Rank4<10, 3, 28, 28>, _>()
            .reshape::<Rank4<10, 3, 28, 28>>();

        let y: Tensor<Rank4<10, 5, 83, 83>, _, _, _> =
            x.leaky_trace().convtrans2d::<3, 2>(w.clone());
        for i in 0..10 {
            assert_close_to_tensor!(y0, y.retaped::<NoneTape>().select(dev.tensor(i)), 1e-5);
        }

        let grads = y.square().mean().backward();

        assert_close_to_tensor!(w0, grads.get(&w));

        let x_grad = grads.get(&x) * 10.0;
        for i in 0..10 {
            assert_close_to_tensor!(x0, x_grad.clone().select(dev.tensor(i)));
        }
    }
}
