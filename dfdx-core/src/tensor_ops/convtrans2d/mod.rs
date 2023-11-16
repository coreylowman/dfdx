mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

#[cfg(test)]
mod tests;

use crate::{shapes::*, tensor::*};

use super::ReshapeTo;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(super) struct ConvTrans2DOp {
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub batch: usize,
    pub chan_in: usize,
    pub chan_out: usize,
    pub h_in: usize,
    pub h_out: usize,
    pub w_in: usize,
    pub w_out: usize,
}

pub(super) trait ConvTrans2DKernel<E: Dtype>: Storage<E> {
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Error>;

    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Error>;

    #[allow(clippy::too_many_arguments)]
    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: ConvTrans2DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

pub trait TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>: Sized {
    type Convolved;

    /// Applies a 2D convolution to the input tensor.
    fn convtrans2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
        output_padding: OutputPadding,
    ) -> Self::Convolved {
        self.try_convtrans2d(stride, padding, dilation, groups, output_padding)
            .unwrap()
    }

    /// Fallibly applies a 2D convolution to the input tensor.
    fn try_convtrans2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
        output_padding: OutputPadding,
    ) -> Result<Self::Convolved, Error>;
}

impl<
        const KERNEL: usize,
        const STRIDE: usize,
        const PADDING: usize,
        const DILATION: usize,
        Groups: Dim,
        const OUTPUT_PADDING: usize,
        const DIM: usize,
    > TryConvTrans2D<Const<STRIDE>, Const<PADDING>, Const<DILATION>, Groups, Const<OUTPUT_PADDING>>
    for (Const<DIM>, Const<KERNEL>)
where
    Const<{ (DIM - 1) * STRIDE - 2 * PADDING + DILATION * (KERNEL - 1) + 1 + OUTPUT_PADDING }>:
        Sized,
{
    type Convolved =
        Const<{ (DIM - 1) * STRIDE - 2 * PADDING + DILATION * (KERNEL - 1) + 1 + OUTPUT_PADDING }>;

    fn try_convtrans2d(
        self,
        _: Const<STRIDE>,
        _: Const<PADDING>,
        _: Const<DILATION>,
        _: Groups,
        _: Const<OUTPUT_PADDING>,
    ) -> Result<Self::Convolved, Error> {
        Ok(Const)
    }
}

impl<Kernel: Dim, Stride: Dim, Padding: Dim, Dilation: Dim, Groups: Dim, OutputPadding: Dim>
    TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding> for (usize, Kernel)
{
    type Convolved = usize;

    fn try_convtrans2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        _: Groups,
        output_padding: OutputPadding,
    ) -> Result<Self::Convolved, Error> {
        let (dim, kernel) = self;
        Ok(((dim - 1) * stride.size()
            + dilation.size() * (kernel.size() - 1)
            + 1
            + output_padding.size())
        .checked_sub(2 * padding.size())
        .unwrap())
    }
}

impl<
        InpChan,
        OutChanOverGroups,
        Kernel,
        Stride,
        Padding,
        Dilation,
        Groups,
        OutputPadding,
        H,
        W,
        E,
        D,
        T,
    > TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>
    for (
        Tensor<(InpChan, H, W), E, D, T>,
        Tensor<(InpChan, OutChanOverGroups, Kernel, Kernel), E, D>,
    )
where
    InpChan: Dim,
    OutChanOverGroups: Dim,
    Kernel: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    OutputPadding: Dim,
    H: Dim,
    W: Dim,
    E: Dtype,
    D: ConvTrans2DKernel<E> + crate::tensor_ops::reshape_to::ReshapeKernel<E>,
    T: Tape<E, D>,
    OutChanOverGroups: std::ops::Mul<Groups>,
    <OutChanOverGroups as std::ops::Mul<Groups>>::Output: Dim,
    (H, Kernel): TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>,
    (W, Kernel): TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>,
    <(H, Kernel) as TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>>::Convolved:
        Dim,
    <(W, Kernel) as TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>>::Convolved:
        Dim,
{
    type Convolved = Tensor<
        (
            <OutChanOverGroups as std::ops::Mul<Groups>>::Output,
            <(H, Kernel) as TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>>::Convolved,
            <(W, Kernel) as TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>>::Convolved,
        ),
        E,
        D,
        T,
    >;

    fn try_convtrans2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
        output_padding: OutputPadding,
    ) -> Result<Self::Convolved, Error> {
        let (img, filters) = self;
        let (inp_chan, h, w) = img.shape;
        let img = img.try_reshape_like(&(Const::<1>, inp_chan, h, w))?;
        let out =
            (img, filters).try_convtrans2d(stride, padding, dilation, groups, output_padding)?;
        let (_, out_chan, out_h, out_w) = out.shape;
        out.try_reshape_like(&(out_chan, out_h, out_w))
    }
}
impl<
        InpChan,
        OutChanOverGroups,
        Kernel,
        Stride,
        Padding,
        Dilation,
        Groups,
        OutputPadding,
        Batch,
        H,
        W,
        E,
        D,
        T,
    > TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>
    for (
        Tensor<(Batch, InpChan, H, W), E, D, T>,
        Tensor<(InpChan, OutChanOverGroups, Kernel, Kernel), E, D>,
    )
where
    InpChan: Dim,
    OutChanOverGroups: Dim,
    Kernel: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    OutputPadding: Dim,
    Batch: Dim,
    H: Dim,
    W: Dim,
    E: Dtype,
    D: ConvTrans2DKernel<E>,
    T: Tape<E, D>,
    OutChanOverGroups: std::ops::Mul<Groups>,
    <OutChanOverGroups as std::ops::Mul<Groups>>::Output: Dim,
    (H, Kernel): TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>,
    (W, Kernel): TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>,
    <(H, Kernel) as TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>>::Convolved:
        Dim,
    <(W, Kernel) as TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>>::Convolved:
        Dim,
{
    type Convolved = Tensor<
        (
            Batch,
            <OutChanOverGroups as std::ops::Mul<Groups>>::Output,
            <(H, Kernel) as TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>>::Convolved,
            <(W, Kernel) as TryConvTrans2D<Stride, Padding, Dilation, Groups, OutputPadding>>::Convolved,
        ),
        E,
        D,
        T,
    >;

    fn try_convtrans2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
        output_padding: OutputPadding,
    ) -> Result<Self::Convolved, Error> {
        let (img, filters) = self;
        assert_eq!(img.shape.1, filters.shape.0);
        assert_eq!(filters.shape.2, filters.shape.3);
        let (batch, _, h, w) = img.shape;
        let (inp_chan, out_chan_over_groups, kernel, _) = filters.shape;
        let out_chan = out_chan_over_groups * groups;
        if img.strides != img.shape.strides() || filters.strides != filters.shape.strides() {
            panic!("Image & filter inputs to conv2d must be contiguous");
        }
        let h_out = (h, kernel).convtrans2d(stride, padding, dilation, groups, output_padding);
        let w_out = (w, kernel).convtrans2d(stride, padding, dilation, groups, output_padding);
        let op = ConvTrans2DOp {
            stride: stride.size(),
            padding: padding.size(),
            kernel: kernel.size(),
            dilation: dilation.size(),
            groups: groups.size(),
            batch: batch.size(),
            chan_in: inp_chan.size(),
            chan_out: out_chan.size(),
            h_in: h.size(),
            h_out: h_out.size(),
            w_in: w.size(),
            w_out: w_out.size(),
        };
        let (lhs, ltape) = img.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut out = lhs.device.alloc((batch, out_chan, h_out, w_out))?;
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
