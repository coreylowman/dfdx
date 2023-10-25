use crate::{shapes::*, tensor::*, tensor_ops::ReshapeTo};

mod cpu_kernel;
#[cfg(all(not(feature = "cudnn"), feature = "cuda"))]
mod cuda_kernel;

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(super) struct Conv1DOp {
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub batch: usize,
    pub chan_in: usize,
    pub chan_out: usize,
    pub l_in: usize,
    pub l_out: usize,
}

pub(super) trait Conv1DKernel<E: Dtype>: Storage<E> {
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Self::Err>;

    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv1DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err>;

    #[allow(clippy::too_many_arguments)]
    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv1DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err>;
}

/// Applies a 1d convolution to a tensor.
///
/// [Const] dims **require nightly**:
/// ```ignore
/// #![feature(generic_const_exprs)]
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<Rank4<2, 3, 32>, f32, _> = dev.sample_normal();
/// let w: Tensor<Rank4<6, 3, 3>, f32, _> = dev.sample_normal();
/// let y = (x, w).conv1d(
///     Const::<1>, // stride
///     Const::<0>, // padding
///     Const::<1>, // dilation
///     Const::<1>, // groups
/// );
/// ```
///
/// [usize] dims can be used on stable:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<_, f32, _> = dev.sample_normal_like(&(
///     2,  // batch size
///     3,  // input channels
///     32, // length
/// ));
/// let w: Tensor<_, f32, _> = dev.sample_normal_like(&(
///     6, // output channels
///     3, // input channels
///     3, // kernel size
/// ));
/// let y = (x, w).conv1d(
///     1, // stride
///     0, // padding
///     1, // dilation
///     1, // groups
/// );
/// ```
pub trait TryConv1D<Stride, Padding, Dilation, Groups>: Sized {
    type Convolved;
    type Error: std::fmt::Debug;

    /// Applies a 1D convolution to the input tensor.
    fn conv1d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
    ) -> Self::Convolved {
        self.try_conv1d(stride, padding, dilation, groups).unwrap()
    }

    /// Fallibly applies a 1D convolution to the input tensor.
    fn try_conv1d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
    ) -> Result<Self::Convolved, Self::Error>;
}

impl<
        const KERNEL: usize,
        const STRIDE: usize,
        const PADDING: usize,
        const DILATION: usize,
        Groups: Dim,
        const DIM: usize,
    > TryConv1D<Const<STRIDE>, Const<PADDING>, Const<DILATION>, Groups>
    for (Const<DIM>, Const<KERNEL>)
where
    Const<{ (DIM + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1 }>: Sized,
{
    type Convolved = Const<{ (DIM + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1 }>;
    type Error = std::convert::Infallible;
    fn try_conv1d(
        self,
        _: Const<STRIDE>,
        _: Const<PADDING>,
        _: Const<DILATION>,
        _: Groups,
    ) -> Result<Self::Convolved, Self::Error> {
        Ok(Const)
    }
}

impl<Kernel: Dim, Stride: Dim, Padding: Dim, Dilation: Dim, Groups: Dim>
    TryConv1D<Stride, Padding, Dilation, Groups> for (usize, Kernel)
{
    type Convolved = usize;
    type Error = std::convert::Infallible;
    fn try_conv1d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        _: Groups,
    ) -> Result<Self::Convolved, Self::Error> {
        let (dim, kernel) = self;
        Ok((dim + 2 * padding.size() - 1)
            .checked_sub(dilation.size() * (kernel.size() - 1))
            .unwrap()
            / stride.size()
            + 1)
    }
}

impl<InpChan, OutChan, Kernel, Stride, Padding, Dilation, Groups, L, E, D, T>
    TryConv1D<Stride, Padding, Dilation, Groups>
    for (
        Tensor<(InpChan, L), E, D, T>,
        Tensor<(OutChan, <InpChan as std::ops::Div<Groups>>::Output, Kernel), E, D>,
    )
where
    InpChan: Dim,
    OutChan: Dim,
    Kernel: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    L: Dim,
    E: Dtype,
    D: Conv1DKernel<E> + crate::tensor_ops::reshape_to::ReshapeKernel<E>,
    T: Tape<E, D>,
    InpChan: std::ops::Div<Groups>,
    <InpChan as std::ops::Div<Groups>>::Output: Dim,
    (L, Kernel): TryConv1D<Stride, Padding, Dilation, Groups>,
    <(L, Kernel) as TryConv1D<Stride, Padding, Dilation, Groups>>::Convolved: Dim,
{
    type Convolved = Tensor<
        (
            OutChan,
            <(L, Kernel) as TryConv1D<Stride, Padding, Dilation, Groups>>::Convolved,
        ),
        E,
        D,
        T,
    >;
    type Error = D::Err;

    fn try_conv1d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
    ) -> Result<Self::Convolved, Self::Error> {
        let (img, filters) = self;
        let (inp_chan, l) = img.shape;
        let img = img.try_reshape_like(&(Const::<1>, inp_chan, l))?;
        let out = (img, filters).try_conv1d(stride, padding, dilation, groups)?;
        let (_, out_chan, out_l) = out.shape;
        out.try_reshape_like(&(out_chan, out_l))
    }
}

impl<InpChan, OutChan, Kernel, Stride, Padding, Dilation, Groups, Batch, L, E, D, T>
    TryConv1D<Stride, Padding, Dilation, Groups>
    for (
        Tensor<(Batch, InpChan, L), E, D, T>,
        Tensor<(OutChan, <InpChan as std::ops::Div<Groups>>::Output, Kernel), E, D>,
    )
where
    InpChan: Dim,
    OutChan: Dim,
    Kernel: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    Batch: Dim,
    L: Dim,
    E: Dtype,
    D: Conv1DKernel<E>,
    T: Tape<E, D>,
    InpChan: std::ops::Div<Groups>,
    <InpChan as std::ops::Div<Groups>>::Output: Dim,
    (L, Kernel): TryConv1D<Stride, Padding, Dilation, Groups>,
    <(L, Kernel) as TryConv1D<Stride, Padding, Dilation, Groups>>::Convolved: Dim,
{
    type Convolved = Tensor<
        (
            Batch,
            OutChan,
            <(L, Kernel) as TryConv1D<Stride, Padding, Dilation, Groups>>::Convolved,
        ),
        E,
        D,
        T,
    >;
    type Error = D::Err;

    fn try_conv1d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
    ) -> Result<Self::Convolved, Self::Error> {
        let (img, filters) = self;
        assert_eq!(img.shape.1.size(), filters.shape.1.size() * groups.size());
        let (batch, inp_chan, l) = img.shape;
        let (out_chan, inp_chan_over_groups, kernel) = filters.shape;
        assert_eq!(inp_chan / groups, inp_chan_over_groups);
        assert!(out_chan.size() % groups.size() == 0);
        if img.strides != img.shape.strides() || filters.strides != filters.shape.strides() {
            panic!("Image & filter inputs to conv1d must be contiguous");
        }
        let l_out = (l, kernel).conv1d(stride, padding, dilation, groups);
        let op = Conv1DOp {
            stride: stride.size(),
            padding: padding.size(),
            kernel: kernel.size(),
            dilation: dilation.size(),
            groups: groups.size(),
            batch: batch.size(),
            chan_in: inp_chan.size(),
            chan_out: out_chan.size(),
            l_in: l.size(),
            l_out: l_out.size(),
        };
        let (lhs, ltape) = img.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut out = lhs.device.alloc((batch, out_chan, l_out))?;
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
