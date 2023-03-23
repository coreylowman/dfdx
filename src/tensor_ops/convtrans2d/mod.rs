mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::*,
    tensor::{DeviceStorage, HasErr, PutTape, SplitTape, Tape, Tensor, ZerosTensor},
};

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
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

pub trait ConvTransAlgebra<const K: usize, const S: usize, const P: usize>: ConstDim {
    type Convolved: ConstDim;
}

impl<const D: usize, const K: usize, const S: usize, const P: usize> ConvTransAlgebra<K, S, P>
    for Const<D>
where
    Const<{ D * S + K - S - 2 * P }>: Sized,
{
    type Convolved = Const<{ D * S + K - S - 2 * P }>;
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

impl<T, F> TryConvTrans2D<F> for T {}

impl<
        const C: usize,
        const H: usize,
        const W: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        E: Dtype,
        D: ConvTrans2DKernel<E> + ZerosTensor<E>,
        T: 'static + Tape<E, D>,
    > TryConvTrans2DTo<Tensor<Rank4<O, C, K, K>, E, D>, S, P> for Tensor<Rank3<C, H, W>, E, D, T>
where
    Const<H>: ConvTransAlgebra<K, S, P>,
    Const<W>: ConvTransAlgebra<K, S, P>,
{
    type Output = Tensor<
        (
            Const<O>,
            <Const<H> as ConvTransAlgebra<K, S, P>>::Convolved,
            <Const<W> as ConvTransAlgebra<K, S, P>>::Convolved,
        ),
        E,
        D,
        T,
    >;

    fn try_convtrans2d_to(
        self,
        filters: Tensor<Rank4<O, C, K, K>, E, D>,
    ) -> Result<Self::Output, Self::Err> {
        let op = ConvTrans2DOp::new(S, P, K, [1, C, H, W], O);
        let (lhs, ltape) = self.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut tape = ltape.merge(rtape);
        let mut out = lhs.device.try_zeros()?;
        lhs.device.forward(op, &lhs, &rhs, &mut out)?;
        let phantom_out = out.clone();
        tape.try_alloc_grad(&lhs)?;
        tape.try_alloc_grad(&rhs)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out);
            lhs.device
                .backward(op, &lhs, grad_lhs, &rhs, grad_rhs, &phantom_out, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

impl<
        B: Dim,
        const C: usize,
        const H: usize,
        const W: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        E: Dtype,
        D: ConvTrans2DKernel<E> + ZerosTensor<E>,
        T: 'static + Tape<E, D>,
    > TryConvTrans2DTo<Tensor<Rank4<O, C, K, K>, E, D>, S, P>
    for Tensor<(B, Const<C>, Const<H>, Const<W>), E, D, T>
where
    Const<H>: ConvTransAlgebra<K, S, P>,
    Const<W>: ConvTransAlgebra<K, S, P>,
{
    type Output = Tensor<
        (
            B,
            Const<O>,
            <Const<H> as ConvTransAlgebra<K, S, P>>::Convolved,
            <Const<W> as ConvTransAlgebra<K, S, P>>::Convolved,
        ),
        E,
        D,
        T,
    >;
    fn try_convtrans2d_to(
        self,
        filters: Tensor<Rank4<O, C, K, K>, E, D>,
    ) -> Result<Self::Output, Self::Err> {
        let batch = self.shape().0;
        let op = ConvTrans2DOp::new(S, P, K, [batch.size(), C, H, W], O);
        let (lhs, ltape) = self.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut out =
            lhs.device
                .try_zeros_like(&(batch, Const, Default::default(), Default::default()))?;
        let mut tape = ltape.merge(rtape);
        lhs.device.forward(op, &lhs, &rhs, &mut out)?;
        let phantom_out = out.clone();
        tape.try_alloc_grad(&lhs)?;
        tape.try_alloc_grad(&rhs)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out);
            lhs.device
                .backward(op, &lhs, grad_lhs, &rhs, grad_rhs, &phantom_out, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor::*, tensor_ops::*, tests::*};

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

        let x = device.tensor(
            [[[0.4918, 0.2600, 0.6637],
         [0.1488, 0.3421, 0.4430],
         [0.8826, 0.5862, 0.0080]],

        [[0.6132, 0.5911, 0.5184],
         [0.2498, 0.1187, 0.0633],
         [0.1825, 0.9979, 0.6280]]]);
        let w = device.tensor([[[[0.4067, 0.0247, 0.5358],
            [0.1852, 0.2280, 0.8278],
            [0.3326, 0.1035, 0.1129]],
  
           [[0.2874, 0.1570, 0.3844],
            [0.5164, 0.4097, 0.2405],
            [0.7282, 0.7366, 0.6046]]],
  
  
          [[[0.3804, 0.1099, 0.3384],
            [0.0087, 0.7392, 0.8201],
            [0.2622, 0.7524, 0.2022]],
  
           [[0.5180, 0.8539, 0.1158],
            [0.6467, 0.4116, 0.8275],
            [0.4141, 0.1176, 0.8865]]],
  
  
          [[[0.8716, 0.4106, 0.9210],
            [0.3776, 0.3710, 0.0988],
            [0.5434, 0.0949, 0.3407]],
  
           [[0.1095, 0.2934, 0.5569],
            [0.7218, 0.1562, 0.4157],
            [0.8636, 0.8561, 0.2443]]]]);
        let y = x.leaky_trace().convtrans2d::<1, 0>(w.clone());
        assert_close(&y.array(), &[[[0.3762, 0.3840, 1.0174, 0.4644, 0.5549],
        [0.5400, 0.9329, 1.6478, 0.9709, 0.9357],
        [1.1780, 1.8561, 2.8093, 2.0724, 1.0160],
        [0.4891, 1.2994, 2.1267, 1.1871, 0.2459],
        [0.4264, 1.1474, 1.4656, 1.1328, 0.3805]],

       [[0.5047, 0.9828, 1.2918, 0.6721, 0.2846],
        [0.5868, 1.4217, 2.1068, 1.6386, 1.1305],
        [0.9760, 2.0403, 3.5367, 2.7203, 1.0849],
        [0.2681, 1.6581, 2.7902, 2.0860, 0.6718],
        [0.3070, 1.2524, 1.1607, 1.0829, 0.5583]],

       [[0.4958, 0.6732, 1.7098, 0.9933, 0.8999],
        [0.7853, 1.2487, 1.9616, 1.1803, 0.7243],
        [1.8225, 2.5686, 3.6571, 2.2803, 0.7799],
        [0.7615, 1.8139, 1.5339, 0.8155, 0.4283],
        [0.6372, 1.4203, 1.8018, 0.9819, 0.1561]]]);

        let g = y.exp().mean().backward();

        assert_close(&g.get(&x).array(), &[[[0.8571, 1.0439, 1.0259],
        [1.2110, 1.4365, 0.9943],
        [1.4347, 1.2419, 1.1606]],

       [[1.3679, 1.5748, 1.4624],
        [1.4738, 1.4895, 1.5150],
        [1.1749, 1.4801, 1.1952]]]);
        assert_close(&g.get(&w).array(), &[[[[0.1748, 0.2835, 0.3342],
        [0.2485, 0.3024, 0.2757],
        [0.3030, 0.3059, 0.2753]],

       [[0.2887, 0.3666, 0.2400],
        [0.2279, 0.2822, 0.2258],
        [0.2919, 0.3576, 0.3157]]],


      [[[0.2184, 0.4842, 0.6237],
        [0.4003, 0.5528, 0.5088],
        [0.5145, 0.5026, 0.4455]],

       [[0.4757, 0.6981, 0.4029],
        [0.3636, 0.5239, 0.4333],
        [0.4228, 0.5644, 0.5513]]],


      [[[0.3076, 0.5719, 0.6412],
        [0.4637, 0.4454, 0.2829],
        [0.5436, 0.4495, 0.4282]],

       [[0.5990, 0.7331, 0.3567],
        [0.2941, 0.3156, 0.2986],
        [0.5524, 0.6224, 0.4893]]]]);
    }
}
