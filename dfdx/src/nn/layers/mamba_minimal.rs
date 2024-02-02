// references:
// - https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/
// - https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/

#![allow(clippy::type_complexity)]

use dfdx::nn::{
    Bias1D, Bias1DConfig, Conv1D, Conv1DConfig, Linear, LinearConfig, MatMul, MatMulConfig,
};
use dfdx::prelude::{
    Axes3, Axes4, Axis, BuildOnDevice, Const, Device, Dim, Dtype, Error, HasShape, Module,
    NoneTape, PutTape, Tape, Tensor,
};
use dfdx::tensor_ops::{
    BroadcastTo, PermuteTo, RealizeTo, ReshapeTo, SumTo, TryAdd, TryConcatTensorAlong, TryMatMul,
    TryMul, TrySplitTensorAlong, TryStack, TryUnstack,
};
#[cfg(feature = "safetensors")]
use dfdx::{LoadSafeTensors, SaveSafeTensors};
use dfdx::{ResetParams, UpdateParams, ZeroGrads};
use std::ops::{Add, Div, Mul, Sub};

pub type C1 = Const<1>;
pub type C2 = Const<2>;
pub type C4 = Const<4>;
pub type C15 = Const<15>;
pub type C16 = Const<16>;

//
/// A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper.
#[derive(Clone, Debug, Default)]
pub struct MambaBlockConfig<
    // Hidden dimension.
    DModel: Dim,
    // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
    //
    // Default: 16
    DState: Dim = C16,
    // Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
    // Δ or delta: input-dependent step size.
    //
    // Default: (DModel + 15) / 16
    DtRank: Dim = <<DModel as Add<C15>>::Output as Div<C16>>::Output,
    // Default: 4
    DConv: Dim = C4,
    // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
    //
    // Default: DModel * 2
    DInner: Dim = <DModel as Mul<C2>>::Output,
> where
    // DModel + 15
    DModel: Add<C15>,
    <DModel as Add<C15>>::Output: Dim,
    // (DModel + 15) / 16
    <DModel as Add<C15>>::Output: Div<C16>,
    <<DModel as Add<C15>>::Output as Div<C16>>::Output: Dim,
    // DModel * 2
    DModel: Mul<C2>,
    <DModel as Mul<C2>>::Output: Dim,
    // DInner * 2
    DInner: Mul<C2>,
    <DInner as Mul<C2>>::Output: Dim + Default,
    // DConv - 1
    DConv: Sub<C1>,
    <DConv as Sub<C1>>::Output: Dim + Default,
    // DState * 2
    DState: Mul<C2>,
    <DState as Mul<C2>>::Output: Dim,
    // DtRank + DState * 2
    DtRank: Add<<DState as Mul<C2>>::Output>,
    <DtRank as Add<<DState as Mul<C2>>::Output>>::Output: Dim + Default,
{
    /// Input: DModel.
    /// Output: DInner * 2.
    pub in_proj: MatMulConfig<DModel, <DInner as Mul<C2>>::Output>,

    /// Input channel: DInner.  
    /// Output channel: DInner.
    pub conv1d: Conv1DConfig<DInner, DInner, DConv, C1, <DConv as Sub<C1>>::Output, C1, DInner>,

    /// Input channel: DInner.  
    /// Output channel: DInner.
    pub conv1d_bias: Bias1DConfig<DInner>,

    /// Takes in the state and outputs the input-specific Δ, B, C.
    ///
    /// Input: DInner.  
    /// Output: DtRank + DState * 2.
    pub x_proj: MatMulConfig<DInner, <DtRank as Add<<DState as Mul<C2>>::Output>>::Output>,

    /// Projects Δ from DT_RANK to D_INNER
    ///
    /// Input: DtRank.  
    /// Output: DInner.
    pub dt_proj: LinearConfig<DtRank, DInner>,

    pub a_log: (DInner, DState),

    pub d: (DInner,),

    // TODO: this could have a bias (becoming a Linear layer)
    // ref: https://github.com/johnma2006/mamba-minimal/blob/03de542a36d873f6e6c4057ad687278cc6ae944d/model.py#L203
    //
    /// Input: DInner.  
    /// Output: DModel.
    pub out_proj: MatMulConfig<DInner, DModel>,
}

//
/// A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper.
pub type MambaBlockConstConfig<
    // Hidden dimension.
    const D_MODEL: usize,
    // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
    const D_STATE: usize = 16,
    const DT_RANK: usize = { (D_MODEL + 15) / 16 },
    const D_CONV: usize = 4,
    const D_INNER: usize = { D_MODEL * 2 },
> = MambaBlockConfig<
    //
    Const<D_MODEL>,
    Const<D_STATE>,
    Const<DT_RANK>,
    Const<D_CONV>,
    Const<D_INNER>,
>;

impl<
        // Hidden dimension.
        DModel: Dim,
        // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
        //
        // Default: 16
        DState: Dim,
        // Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
        // Δ or delta: input-dependent step size.
        //
        // Default: (DModel + 15) / 16
        DtRank: Dim,
        // Default: 4
        DConv: Dim,
        // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
        //
        // Default: DModel * 2
        DInner: Dim,
    > MambaBlockConfig<DModel, DState, DtRank, DConv, DInner>
where
    // DModel + 15
    DModel: Add<C15>,
    <DModel as Add<C15>>::Output: Dim,
    // (DModel + 15) / 16
    <DModel as Add<C15>>::Output: Div<C16>,
    <<DModel as Add<C15>>::Output as Div<C16>>::Output: Dim,
    // DModel * 2
    DModel: Mul<C2>,
    <DModel as Mul<C2>>::Output: Dim,
    // DInner * 2
    DInner: Mul<C2>,
    <DInner as Mul<C2>>::Output: Dim + Default,
    // DConv - 1
    DConv: Sub<C1>,
    <DConv as Sub<C1>>::Output: Dim + Default,
    // DState * 2
    DState: Mul<C2>,
    <DState as Mul<C2>>::Output: Dim,
    // DtRank + DState * 2
    DtRank: Add<<DState as Mul<C2>>::Output>,
    <DtRank as Add<<DState as Mul<C2>>::Output>>::Output: Dim + Default,
{
    pub fn new(
        d_model: DModel,
        d_state: DState,
        dt_rank: DtRank,
        d_conv: DConv,
        d_inner: DInner,
    ) -> Self {
        MambaBlockConfig {
            in_proj: MatMulConfig {
                inp: d_model,
                out: d_inner * Const::<2>,
            },
            conv1d: Conv1DConfig {
                in_chan: d_inner,
                out_chan: d_inner,
                kernel_size: d_conv,
                stride: Const::<1>,
                padding: d_conv - Const::<1>,
                dilation: Const::<1>,
                groups: d_inner,
            },
            conv1d_bias: Bias1DConfig(d_inner),
            x_proj: MatMulConfig {
                inp: d_inner,
                out: dt_rank + d_state * Const::<2>,
            },
            dt_proj: LinearConfig {
                inp: dt_rank,
                out: d_inner,
            },
            a_log: (d_inner, d_state),
            d: (d_inner,),
            out_proj: MatMulConfig {
                inp: d_inner,
                out: d_model,
            },
        }
    }
}

//
/// A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper.
#[derive(Clone, Debug, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct MambaBlock<
    // Hidden dimension.
    DModel: Dim,
    // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
    DState: Dim,
    // Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
    // Δ or delta: input-dependent step size.
    DtRank: Dim,
    DConv: Dim,
    // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
    // By default, expand is implicitly `2`.
    DInner: Dim,
    Elem: Dtype,
    Dev: Device<Elem>,
> where
    // DInner can be divided by itself
    DInner: Div<DInner>,
    <DInner as Div<DInner>>::Output: Dim,
    // DInner * 2
    DInner: Mul<C2>,
    <DInner as Mul<C2>>::Output: Dim,
    // DConv - 1
    DConv: Sub<C1>,
    <DConv as Sub<C1>>::Output: Dim + Default,
    // DState * 2
    DState: Mul<C2>,
    <DState as Mul<C2>>::Output: Dim,
    // DtRank + DState * 2
    DtRank: Add<<DState as Mul<C2>>::Output>,
    <DtRank as Add<<DState as Mul<C2>>::Output>>::Output: Dim + Default,
{
    /// Input: DModel.
    /// Output: DInner * 2.
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub in_proj: MatMul<DModel, <DInner as Mul<C2>>::Output, Elem, Dev>,

    // TODO: is the padding correct? (DConv - 1)
    // is it different in here?
    // https://github.com/kroggen/mamba-cpu/blob/d12b23b059d249b7077ad080679ae918c9a45caf/mamba_ssm/modules/mamba_simple.py#L103
    //
    /// Input channel: DInner.
    /// Output channel: DInner.
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub conv1d:
        Conv1D<DInner, DInner, DConv, C1, <DConv as Sub<C1>>::Output, C1, DInner, Elem, Dev>,

    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub conv1d_bias: Bias1D<DInner, Elem, Dev>,

    /// Takes in the state and outputs the input-specific Δ, B, C.
    ///
    /// Input: DInner.
    /// Output: DtRank + DState * 2.
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub x_proj: MatMul<DInner, <DtRank as Add<<DState as Mul<C2>>::Output>>::Output, Elem, Dev>,

    /// Projects Δ.
    ///
    /// Input: DtRank.
    /// Output: DInner.
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub dt_proj: Linear<DtRank, DInner, Elem, Dev>,

    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub a_log: Tensor<(DInner, DState), Elem, Dev>,

    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub d: Tensor<(DInner,), Elem, Dev>,

    // TODO: this could have a bias (becoming a Linear layer)
    // ref: https://github.com/johnma2006/mamba-minimal/blob/03de542a36d873f6e6c4057ad687278cc6ae944d/model.py#L203
    /// Input: DInner.
    /// Output: DModel.
    #[module]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub out_proj: MatMul<DInner, DModel, Elem, Dev>,
}

impl<DModel: Dim, DState: Dim, DtRank: Dim, DConv: Dim, DInner: Dim, E: Dtype, D: Device<E>>
    BuildOnDevice<E, D> for MambaBlockConfig<DModel, DState, DtRank, DConv, DInner>
where
    // DModel + 15
    DModel: Add<C15>,
    <DModel as Add<C15>>::Output: Dim,
    // (DModel + 15) / 16
    <DModel as Add<C15>>::Output: Div<C16>,
    <<DModel as Add<C15>>::Output as Div<C16>>::Output: Dim,
    // DModel * 2
    DModel: Mul<C2>,
    <DModel as Mul<C2>>::Output: Dim,
    // DInner can be divided by itself
    DInner: Div<DInner>,
    <DInner as Div<DInner>>::Output: Dim,
    // DInner * 2
    DInner: Mul<C2>,
    <DInner as Mul<C2>>::Output: Dim + Default,
    // DConv - 1
    DConv: Sub<C1>,
    <DConv as Sub<C1>>::Output: Dim + Default,
    // DState * 2
    DState: Mul<C2>,
    <DState as Mul<C2>>::Output: Dim,
    // DtRank + DState * 2
    DtRank: Add<<DState as Mul<C2>>::Output>,
    <DtRank as Add<<DState as Mul<C2>>::Output>>::Output: Dim + Default,
{
    type Built = MambaBlock<DModel, DState, DtRank, DConv, DInner, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, dfdx::tensor::Error> {
        Ok(MambaBlock {
            in_proj: self.in_proj.try_build_on_device(device)?,
            conv1d: self.conv1d.try_build_on_device(device)?,
            conv1d_bias: self.conv1d_bias.try_build_on_device(device)?,
            x_proj: self.x_proj.try_build_on_device(device)?,
            dt_proj: self.dt_proj.try_build_on_device(device)?,
            a_log: device.try_zeros_like(&self.a_log)?,
            d: device.try_zeros_like(&self.d)?,
            out_proj: self.out_proj.try_build_on_device(device)?,
        })
    }
}

pub mod stateless {
    use super::*;

    #[allow(clippy::let_unit_value)]
    impl<
            // Batch size (`B` in Algorithm 2 from the Mamba paper).
            Batch: Dim,
            // Sequence length (`L` in Algorithm 2 from the Mamba paper).
            Sequence: Dim,
            // Hidden dimension.
            DModel: Dim,
            // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
            DState: Dim,
            // Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
            // Δ or delta: input-dependent step size.
            DtRank: Dim,
            DConv: Dim,
            // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
            DInner: Dim,
            E: Dtype,
            D: Device<E>,
            T: Tape<E, D>,
        > Module<Tensor<(Batch, Sequence, DModel), E, D, T>>
        for MambaBlock<DModel, DState, DtRank, DConv, DInner, E, D>
    where
        // DInner can be divided by itself
        DInner: Div<DInner>,
        <DInner as Div<DInner>>::Output: Dim,
        // DInner * 2
        DInner: Mul<C2>,
        <DInner as Mul<C2>>::Output: Dim,
        // DInner * 2 / 2 = DInner
        <DInner as Mul<C2>>::Output: Div<C2, Output = DInner>,
        // DConv - 1
        DConv: Sub<C1>,
        <DConv as Sub<C1>>::Output: Dim + Default,
        // DState * 2
        DState: Mul<C2>,
        <DState as Mul<C2>>::Output: Dim,
        // DtRank + DState * 2
        DtRank: Add<<DState as Mul<C2>>::Output>,
        <DtRank as Add<<DState as Mul<C2>>::Output>>::Output: Dim + Default,
        // layer 2 (conv1d)
        // used to truncate back to Sequence: Sequence + DConv
        Sequence: Add<DConv>,
        <Sequence as Add<DConv>>::Output: Dim,
        // used to truncate back to Sequence: Sequencec + DConv - 1
        <Sequence as Add<DConv>>::Output: Sub<C1>,
        <<Sequence as Add<DConv>>::Output as Sub<C1>>::Output: Dim,
        Conv1D<
            // in channel
            DInner,
            // out chanel
            DInner,
            // kernel
            DConv,
            // stride
            C1,
            // padding = DConv - 1
            <DConv as Sub<C1>>::Output,
            // dillation
            C1,
            // groups
            DInner,
            E,
            D,
        >: Module<
            Tensor<(Batch, DInner, Sequence), E, D, T>,
            Output = Tensor<
                // (Batch, DInner, Sequence + DConv - 1)
                // but this is later truncated back to (Batch, DInner, Sequence)
                (
                    Batch,
                    DInner,
                    <<Sequence as Add<DConv>>::Output as Sub<C1>>::Output,
                ),
                E,
                D,
                T,
            >,
        >,
        // conv1d bias
        Bias1D<DInner, E, D>: Module<
            Tensor<(Batch, DInner, Sequence), E, D, T>,
            Output = Tensor<(Batch, DInner, Sequence), E, D, T>,
        >,
        // dt_proj bias
        // (this needs to be defined otherwise Rust thinks this should behave the same as conv1d bias)
        Bias1D<DInner, E, D>: Module<
            Tensor<(Batch, Sequence, DtRank), E, D, T>,
            Output = Tensor<(Batch, Sequence, DtRank), E, D, T>,
        >,
    {
        type Output = Tensor<(Batch, Sequence, DModel), E, D, T>;

        //
        /// Mamba block forward.
        /// This looks the same as Figure 3 in Section 3.4 in the Mamba paper.
        fn try_forward(
            &self,
            x: Tensor<(Batch, Sequence, DModel), E, D, T>,
        ) -> Result<Self::Output, Error> {
            let (batch, sequence, _d_model) = *x.shape();
            let (d_inner,) = *self.d.shape();

            // layer 1 (in_proj)
            let (xs, res): (
                Tensor<(Batch, Sequence, DInner), _, _, _>,
                Tensor<(Batch, Sequence, DInner), _, _, _>,
            ) = {
                // projects the input DModel into 2*DInner
                let xs_and_res: Tensor<(Batch, Sequence, <DInner as Mul<C2>>::Output), _, _, _> =
                    self.in_proj.try_forward(x)?;

                // splits xs_and_res into (xs, res)
                let (xs, res, _tape) =
                    xs_and_res.try_split_tensor_along(Axis::<2>, d_inner, d_inner)?;

                (xs, res)
            };

            // layer 2 (conv1d)
            let xs: Tensor<(Batch, Sequence, DInner), _, _, _> = {
                let xs: Tensor<(Batch, DInner, Sequence), _, _, _> =
                    xs.try_permute::<_, Axes3<0, 2, 1>>()?;
                let xs: Tensor<(Batch, DInner, _), _, _, _> =
                    self.conv1d.try_forward(xs.try_contiguous()?)?;

                // assert shape
                {
                    let (_, _, d_conv) = self.conv1d.weight.shape();
                    let xs_shape = xs.shape();
                    debug_assert_eq!(
                        (
                            batch.size(),
                            d_inner.size(),
                            sequence.size() + d_conv.size() - 1
                        ),
                        (xs_shape.0.size(), xs_shape.1.size(), xs_shape.2.size())
                    );
                }

                // make the last axis be limited to the size of 0..sequence
                let (_d_inner, _, d_conv) = *self.conv1d.weight.shape();
                let (xs, _tail, _tape): (Tensor<(Batch, DInner, Sequence), _, _, _>, _, _) =
                    xs.try_split_tensor_along(Axis::<2>, sequence, d_conv - Const::<1>)?;

                // conv1d bias, and restore original positioning as per before the layer 2
                let xs: Tensor<(Batch, Sequence, DInner), _, _, _> =
                    xs.try_permute::<_, Axes3<0, 2, 1>>()?;
                let xs = self.conv1d_bias.try_forward(xs)?;

                // activation
                xs.try_silu()?
            };

            let ss = ss(
                self.a_log.retaped::<T>(),
                self.d.retaped::<T>(),
                xs,
                &self.x_proj,
                &self.dt_proj,
            )?;

            let ys = ss.try_mul(res.try_silu()?)?;

            let y: Tensor<(Batch, Sequence, DModel), _, _, _> = self.out_proj.try_forward(ys)?;
            Ok(y)
        }
    }

    /// Runs the SSM. See:
    /// - Algorithm 2 in Section 3.2 from the Mamba paper;
    /// - run_SSM(A, B, C, u) from The Annotated S4.
    ///
    pub fn ss<
        // Batch size (`B` in Algorithm 2 from the Mamba paper).
        Batch: Dim,
        // Sequence length (`L` in Algorithm 2 from the Mamba paper).
        Sequence: Dim,
        // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
        DState: Dim,
        // Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
        // Δ or delta: input-dependent step size.
        DtRank: Dim,
        // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
        DInner: Dim,
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
    >(
        a: Tensor<(DInner, DState), E, D, T>,
        d: Tensor<(DInner,), E, D, T>,
        u: Tensor<(Batch, Sequence, DInner), E, D, T>,
        x_proj: &MatMul<DInner, <DtRank as Add<<DState as Mul<C2>>::Output>>::Output, E, D>,
        dt_proj: &Linear<DtRank, DInner, E, D>,
    ) -> Result<Tensor<(Batch, Sequence, DInner), E, D, T>, dfdx::tensor::Error>
    where
        // used to truncate back to DtRank: DState * 2
        DState: Mul<C2>,
        <DState as Mul<C2>>::Output: Dim,
        // used to truncate back to DtRank: DtRank + DState * 2
        DtRank: Add<<DState as Mul<C2>>::Output>,
        <DtRank as Add<<DState as Mul<C2>>::Output>>::Output: Dim + Default,
    {
        let device = u.device().clone();

        let (_d_inner, d_state) = *a.shape();
        let (_d_inner, dt_rank) = *dt_proj.weight.shape();

        // Compute ∆ A B C D, the state space parameters.

        // A
        // this is input independent (see Section 3.5.2 "Interpretation of A" form the Mamba paper for why A isn't selective)
        let a: Tensor<(DInner, DState), _, _, _> = a.try_exp()?.try_negate()?;

        // (Batch, Sequence, DtRank + DState * 2)
        let x_dbl: Tensor<(Batch, Sequence, _), _, _, _> = x_proj.try_forward(u.retaped::<T>())?;

        // ∆ (part 1/2)
        // ∆ is input-dependent
        let (delta, x_dbl_tail, _tape): (Tensor<(Batch, Sequence, DtRank), _, _, _>, _, _) =
            x_dbl.try_split_tensor_along(Axis::<2>, dt_rank, d_state * Const::<2>)?;

        // B and C
        // B and C are input-dependent
        let (b, c, _tape): (
            Tensor<(Batch, Sequence, DState), _, _, _>,
            Tensor<(Batch, Sequence, DState), _, _, _>,
            _,
        ) = x_dbl_tail.try_split_tensor_along(Axis::<2>, d_state, d_state)?;

        // ∆ (part 2/2)
        // ∆ is input-dependent
        let delta: Tensor<(Batch, Sequence, DInner), _, _, _> = {
            let delta = dt_proj.try_forward(delta)?;
            // softplus without threshold
            // TODO: consider the threshold
            let one = device.ones_like(&delta);
            (delta.try_exp()?.try_add(one)?).try_ln()?
        };

        selective_scan(
            delta.try_permute::<_, Axes3<0, 2, 1>>()?,
            a,
            b,
            c.try_permute::<_, Axes3<1, 0, 2>>()?,
            d,
            u,
        )
    }

    /// Selective Scan.
    ///
    /// Does selective scan algorithm. See:
    /// - Section 2 State Space Models from the Mamba paper;
    /// - Algorithm 2 in Section 3.2 from the Mamba paper;
    /// - run_SSM(A, B, C, u) from The Annotated S4.
    pub fn selective_scan<
        // Batch size (`B` in Algorithm 2 from the Mamba paper).
        Batch: Dim,
        // Sequence length (`L` in Algorithm 2 from the Mamba paper).
        Sequence: Dim,
        // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
        DState: Dim,
        // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
        DInner: Dim,
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
    >(
        delta: Tensor<(Batch, DInner, Sequence), E, D, T>,
        a: Tensor<(DInner, DState), E, D, T>,
        b: Tensor<(Batch, Sequence, DState), E, D, T>,
        c: Tensor<(Sequence, Batch, DState), E, D, T>,
        d: Tensor<(DInner,), E, D, T>,
        u: Tensor<(Batch, Sequence, DInner), E, D, T>,
    ) -> Result<Tensor<(Batch, Sequence, DInner), E, D, T>, dfdx::tensor::Error> {
        let device = delta.device().clone();

        let (batch, d_inner, sequence) = *delta.shape();
        let (_d_inner, d_state) = *a.shape();

        // Discretize continuous parameters (A, B)
        //  - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper)
        //  - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        //    "A is the more important term and the performance doesn't change much with the simplification on B"
        let (delta_a, delta_bu): (
            Tensor<(Batch, DInner, Sequence, DState), _, _, _>,
            Tensor<(Batch, DInner, Sequence, DState), _, _, _>,
        ) = {
            let target_shape = (batch, d_inner, sequence, d_state);

            let delta_shape = delta.try_broadcast_like(&target_shape)?;

            let a = a.try_broadcast_like(&target_shape)?;
            let delta_a: Tensor<(Batch, DInner, Sequence, DState), _, _, _> =
                delta_shape.retaped::<T>().try_mul(a)?.try_exp()?;

            let b = b.try_broadcast_like(&target_shape)?;
            let delta_bu = delta_shape.try_mul(b)?;

            let u_bu = u
                .retaped::<T>()
                .try_permute::<_, Axes3<0, 2, 1>>()?
                .try_broadcast_like(&target_shape)?;
            let delta_bu = delta_bu.try_mul(u_bu)?;

            (delta_a, delta_bu)
        };

        // Perform selective scan (see scan_SSM() from The Annotated S4)
        // Note that the below is sequential, while the official implementation does a much faster parallel scan that
        // is additionally hardware-aware (like FlashAttention).

        let mut xs: Tensor<(Batch, DInner, DState), E, _, _> = device
            .zeros_like(&(batch, d_inner, d_state))
            .put_tape(T::default());
        let mut ys: Vec<Tensor<(Batch, DInner), _, _, _>> = Vec::with_capacity(sequence.size());

        // permute so that the Sequence refers to the first axis
        let delta_a: Tensor<(Sequence, Batch, DInner, DState), _, _, _> =
            delta_a.try_permute::<_, Axes4<2, 0, 1, 3>>()?;
        let delta_bu: Tensor<(Sequence, Batch, DInner, DState), _, _, _> =
            delta_bu.try_permute::<_, Axes4<2, 0, 1, 3>>()?;

        // unstack the Sequence axis
        //
        // delta A
        let delta_a: Tensor<(usize, Batch, DInner, DState), _, _, _> = match delta_a.try_realize() {
            Ok(delta_a) => delta_a,
            Err(_delta_a) => unreachable!(),
        };
        let (delta_a, _delta_a_tape): (Vec<Tensor<(Batch, DInner, DState), _, _, _>>, _) =
            delta_a.try_contiguous()?.try_unstack()?;
        //
        // delta B
        let delta_bu: Tensor<(usize, Batch, DInner, DState), _, _, _> = match delta_bu.try_realize()
        {
            Ok(delta_bu) => delta_bu,
            Err(_delta_bu) => unreachable!(),
        };
        let (delta_bu, _delta_bu_tape): (Vec<Tensor<(Batch, DInner, DState), _, _, _>>, _) =
            delta_bu.try_contiguous()?.try_unstack()?;
        //
        // C
        let c: Tensor<(usize, Batch, DState, C1), _, _, _> = match c
            .try_broadcast_like(&(sequence, batch, d_state, Const::<1>))?
            .try_realize()
        {
            Ok(c) => c,
            Err(_c) => unreachable!(),
        };
        let (c, _c_tape): (Vec<Tensor<(Batch, DState, C1), _, _, _>>, _) = c.try_unstack()?;

        // loop over the sequence
        for ((delta_a, delta_bu), c) in delta_a
            .into_iter()
            .zip(delta_bu.into_iter())
            .zip(c.into_iter())
        {
            xs = xs.retaped::<T>().try_mul(delta_a)?.try_add(delta_bu)?;
            let y: Tensor<(Batch, DInner), _, _, _> = xs
                .retaped::<T>()
                .try_matmul(c)?
                .try_reshape_like(&(batch, d_inner))?;
            ys.push(y);
        }

        let ys: Tensor<(Batch, Sequence, DInner), _, _, _> = if let Ok(ys) = ys
            .try_stack()?
            .try_permute::<_, Axes3<1, 0, 2>>()?
            .try_realize::<(Batch, Sequence, DInner)>()
        {
            ys
        } else {
            // TODO
            // try_realize whould never fail in this case?
            todo!();
        };

        // D
        let d: Tensor<(Batch, Sequence, DInner), _, _, T> =
            d.try_broadcast_like(&(batch, sequence, d_inner))?;
        let u = u;
        let du = d.try_mul(u)?;

        let ys: Tensor<(Batch, Sequence, DInner), _, _, _> = ys.try_add(du)?;
        Ok(ys)
    }
}

pub mod stateful {
    // additional references:
    // - https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    // - https://github.com/kroggen/mamba.c/blob/learning/mamba.c
    // - https://github.com/kroggen/mamba-cpu/blob/recurrent-only/mamba_ssm/mamba_simple.py

    use super::*;

    #[derive(Clone, Debug, ResetParams)]
    pub struct MambaStateCacheConfig<
        Batch: Dim,
        // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
        DState: Dim,
        DConv: Dim,
        // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
        DInner: Dim,
    > {
        pub conv_state: (Batch, DInner, DConv),
        pub ssm_state: (Batch, DInner, DState),
    }

    #[derive(Debug, Clone, ResetParams)]
    pub struct MambaStateCache<
        Batch: Dim,
        // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
        DState: Dim,
        DConv: Dim,
        // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
        DInner: Dim,
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
    > {
        pub conv_state: Tensor<(Batch, DInner, DConv), E, D, T>,
        pub ssm_state: Tensor<(Batch, DInner, DState), E, D, T>,
    }

    impl<Batch: Dim, DState: Dim, DConv: Dim, DInner: Dim>
        MambaStateCacheConfig<Batch, DState, DConv, DInner>
    {
        pub fn new(batch: Batch, d_state: DState, d_conv: DConv, d_inner: DInner) -> Self {
            Self {
                conv_state: (batch, d_inner, d_conv),
                ssm_state: (batch, d_inner, d_state),
            }
        }
    }

    impl<Batch: Dim, DState: Dim, DConv: Dim, DInner: Dim, E: Dtype, D: Device<E>>
        BuildOnDevice<E, D> for MambaStateCacheConfig<Batch, DState, DConv, DInner>
    {
        type Built = MambaStateCache<Batch, DState, DConv, DInner, E, D, NoneTape>;
        fn try_build_on_device(&self, device: &D) -> Result<Self::Built, dfdx::tensor::Error> {
            Ok(MambaStateCache {
                conv_state: device.try_zeros_like(&self.conv_state)?,
                ssm_state: device.try_zeros_like(&self.ssm_state)?,
            })
        }
    }

    #[allow(clippy::let_unit_value)]
    impl<
            // Batch size (`B` in Algorithm 2 from the Mamba paper).
            Batch: Dim,
            // Hidden dimension.
            DModel: Dim,
            // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
            DState: Dim,
            // Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
            // Δ or delta: input-dependent step size.
            DtRank: Dim,
            DConv: Dim,
            // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
            DInner: Dim,
            E: Dtype,
            D: Device<E>,
            T: Tape<E, D>,
        >
        Module<(
            Tensor<(Batch, DModel), E, D, T>,
            MambaStateCache<Batch, DState, DConv, DInner, E, D, T>,
        )> for MambaBlock<DModel, DState, DtRank, DConv, DInner, E, D>
    where
        // DInner can be divided by itself
        DInner: Div<DInner>,
        <DInner as Div<DInner>>::Output: Dim,
        // DInner * 2
        DInner: Mul<C2>,
        <DInner as Mul<C2>>::Output: Dim,
        // DInner * 2 / 2 = DInner
        <DInner as Mul<C2>>::Output: Div<C2, Output = DInner>,
        // DConv - 1
        DConv: Sub<C1>,
        <DConv as Sub<C1>>::Output: Dim + Default,
        // DState * 2
        DState: Mul<C2>,
        <DState as Mul<C2>>::Output: Dim,
        // DtRank + DState * 2
        DtRank: Add<<DState as Mul<C2>>::Output>,
        <DtRank as Add<<DState as Mul<C2>>::Output>>::Output: Dim + Default,
        // layer 2 (conv1d)
        (
            (
                Batch,
                DInner,
                <DConv as Sub<dfdx_core::shapes::Const<1>>>::Output,
            ),
            (Batch, DInner, dfdx_core::shapes::Const<1>),
        ): dfdx_core::tensor_ops::TryConcatShapeAlong<Axis<2>, Output = (Batch, DInner, DConv)>,
    {
        type Output = (
            Tensor<(Batch, DModel), E, D, T>,
            MambaStateCache<Batch, DState, DConv, DInner, E, D, T>,
        );

        /// Mamba block forward.
        fn try_forward(
            &self,
            x: (
                Tensor<(Batch, DModel), E, D, T>,
                MambaStateCache<Batch, DState, DConv, DInner, E, D, T>,
            ),
        ) -> Result<Self::Output, Error> {
            let (x, mut cache) = x;

            let (batch, d_inner, d_conv) = *cache.conv_state.shape();

            // layer 1 (in_proj)
            let (xs, res): (
                Tensor<(Batch, DInner), _, _, _>,
                Tensor<(Batch, DInner), _, _, _>,
            ) = {
                // projects the input DModel into 2*DInner
                let xs_and_res: Tensor<(Batch, <DInner as Mul<C2>>::Output), _, _, _> =
                    self.in_proj.try_forward(x)?;

                // splits xs_and_res into (xs, res)
                let (xs, res, _tape) =
                    xs_and_res.try_split_tensor_along(Axis::<1>, d_inner, d_inner)?;

                (xs, res)
            };

            // layer 2 (conv1d)
            //
            // needs to replace the first column of cache.conv_state with
            // the new input and roll it so it's the last column
            cache.conv_state = {
                // not sure if there is a way to directly replace just a single column,
                // so the workaround is first to split away the first column (by the left side)
                let (_head, conv_state, _tape): (
                    _,
                    Tensor<(Batch, DInner, <DConv as Sub<C1>>::Output), _, _, _>,
                    _,
                ) = cache.conv_state.try_split_tensor_along(
                    Axis::<2>,
                    Const::<1>,
                    d_conv - Const::<1>,
                )?;
                // then concat with the xs as the last column (by the right side)
                let xs: Tensor<(Batch, DInner, C1), _, _, _> =
                    xs.try_reshape_like(&(batch, d_inner, Const::<1>))?;
                (conv_state, xs).try_concat_tensor_along(Axis::<2>)?
            };

            let xs: Tensor<(Batch, DInner), E, _, _> = {
                let conv1d = self
                    .conv1d
                    .weight
                    .clone()
                    .try_reshape_like(&(d_inner, d_conv))?
                    .try_broadcast_like(&(batch, d_inner, d_conv))?;
                let xs: Tensor<(Batch, DInner, DConv), _, _, _> =
                    cache.conv_state.retaped::<T>().try_mul(conv1d)?;
                let xs: Tensor<(Batch, DInner), _, _, _> = xs.try_sum::<_, Axis<2>>()?;

                // conv1d bias
                let xs = self.conv1d_bias.try_forward(xs)?;

                // activation
                xs.try_silu()?
            };

            let (ss, cache_ssm_state) = ss_step::<Batch, DState, DtRank, DInner, E, D, T>(
                //
                self.a_log.retaped::<T>(),
                self.d.retaped::<T>(),
                xs,
                &self.x_proj,
                &self.dt_proj,
                cache.ssm_state,
            )?;

            let ys = ss.try_mul(res.try_silu()?)?;
            let y: Tensor<(Batch, DModel), _, _, _> = self.out_proj.try_forward(ys)?;

            cache.ssm_state = cache_ssm_state;

            Ok((y, cache))
        }
    }

    /// Runs the SSM. See:
    /// - Algorithm 2 in Section 3.2 from the Mamba paper;
    /// - run_SSM(A, B, C, u) from The Annotated S4.
    pub fn ss_step<
        // Batch size (`B` in Algorithm 2 from the Mamba paper).
        Batch: Dim,
        // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
        DState: Dim,
        // Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
        // Δ or delta: input-dependent step size.
        DtRank: Dim,
        // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
        DInner: Dim,
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
    >(
        //
        a: Tensor<(DInner, DState), E, D, T>,
        d: Tensor<(DInner,), E, D, T>,
        u: Tensor<(Batch, DInner), E, D, T>,
        x_proj: &MatMul<DInner, <DtRank as Add<<DState as Mul<C2>>::Output>>::Output, E, D>,
        dt_proj: &Linear<DtRank, DInner, E, D>,
        ssm_state_cache: Tensor<(Batch, DInner, DState), E, D, T>,
    ) -> Result<
        (
            Tensor<(Batch, DInner), E, D, T>,
            Tensor<(Batch, DInner, DState), E, D, T>,
        ),
        dfdx::tensor::Error,
    >
    where
        // used to truncate back to DtRank: DState * 2
        DState: Mul<C2>,
        <DState as Mul<C2>>::Output: Dim,
        // used to truncate back to DtRank: DtRank + DState * 2
        DtRank: Add<<DState as Mul<C2>>::Output>,
        <DtRank as Add<<DState as Mul<C2>>::Output>>::Output: Dim + Default,
    {
        let device = u.device().clone();

        let (_d_inner, dt_rank) = *dt_proj.weight.shape();
        let (batch, d_inner, d_state) = *ssm_state_cache.shape();

        // Compute ∆ A B C D, the state space parameters.

        // A
        // this is input independent (see Section 3.5.2 "Interpretation of A" form the Mamba paper for why A isn't selective)
        let a: Tensor<(DInner, DState), _, _, _> = a.try_exp()?.try_negate()?;

        // (Batch, DtRank + DState * 2)
        let x_dbl: Tensor<(Batch, _), _, _, _> = x_proj.try_forward(u.retaped::<T>())?;

        // ∆ (part 1/2)
        // ∆ is input-dependent
        let (delta, x_dbl_tail, _tape): (Tensor<(Batch, DtRank), _, _, _>, _, _) =
            x_dbl.try_split_tensor_along(Axis::<1>, dt_rank, d_state * Const::<2>)?;

        // B and C
        // B and C are input-dependent
        let (b, c, _tape): (
            Tensor<(Batch, DState), _, _, _>,
            Tensor<(Batch, DState), _, _, _>,
            _,
        ) = x_dbl_tail.try_split_tensor_along(Axis::<1>, d_state, d_state)?;

        // ∆ (part 2/2)
        // ∆ is input-dependent
        let delta: Tensor<(Batch, DInner), _, _, _> = {
            // note: don't add dt_proj bias
            let delta = delta.try_matmul(
                dt_proj
                    .weight
                    .retaped::<T>()
                    .try_permute::<_, dfdx::prelude::Axes2<1, 0>>()?,
            )?;
            // softplus without threshold
            // TODO: consider the threshold
            let one = device.ones_like(&delta);
            (delta
                .try_add(
                    dt_proj
                        .bias
                        .retaped::<T>()
                        .try_broadcast_like(&(batch, d_inner))?,
                )?
                .try_exp()?
                .try_add(one)?)
            .try_ln()?
        };

        selective_scan_step::<Batch, DState, DInner, E, D, T>(delta, a, b, c, d, u, ssm_state_cache)
    }

    // Selective Scan.
    ///
    /// Does selective scan algorithm. See:
    /// - Section 2 State Space Models from the Mamba paper;
    /// - Algorithm 2 in Section 3.2 from the Mamba paper;
    /// - run_SSM(A, B, C, u) from The Annotated S4.
    ///
    pub fn selective_scan_step<
        // Batch size (`B` in Algorithm 2 from the Mamba paper).
        Batch: Dim,
        // latent state dimension (`N` in Algorithm 2 from the Mamba paper).
        DState: Dim,
        // DModel * expand (`D` in Algorithm 2 from the Mamba paper).
        DInner: Dim,
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
    >(
        delta: Tensor<(Batch, DInner), E, D, T>,
        a: Tensor<(DInner, DState), E, D, T>,
        b: Tensor<(Batch, DState), E, D, T>,
        c: Tensor<(Batch, DState), E, D, T>,
        d: Tensor<(DInner,), E, D, T>,
        u: Tensor<(Batch, DInner), E, D, T>,
        mut ssm_state_cache: Tensor<(Batch, DInner, DState), E, D, T>,
    ) -> Result<
        (
            Tensor<(Batch, DInner), E, D, T>,
            Tensor<(Batch, DInner, DState), E, D, T>,
        ),
        dfdx::tensor::Error,
    > {
        let (batch, d_inner, d_state) = *ssm_state_cache.shape();

        // Discretize continuous parameters (A, B)
        //  - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper)
        //  - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        //    "A is the more important term and the performance doesn't change much with the simplification on B"
        let (delta_a, delta_bu): (
            Tensor<(Batch, DInner, DState), _, _, _>,
            Tensor<(Batch, DInner, DState), _, _, _>,
        ) = {
            let target_shape = (batch, d_inner, d_state);

            let delta_broadcasted = delta.try_broadcast_like(&target_shape)?;

            let a = a.try_broadcast_like(&target_shape)?;
            let delta_a: Tensor<(Batch, DInner, DState), _, _, _> =
                delta_broadcasted.retaped::<T>().try_mul(a)?.try_exp()?;

            let b = b.try_broadcast_like(&target_shape)?;
            let delta_bu = delta_broadcasted.try_mul(b)?;

            (delta_a, delta_bu)
        };

        ssm_state_cache = ssm_state_cache
            .try_mul(delta_a.try_reshape_like(&(batch, d_inner, d_state))?)?
            .try_add(
                u.retaped::<T>()
                    .try_reshape_like(&(batch, d_inner))?
                    .try_broadcast_like(&(batch, d_inner, d_state))?
                    .try_mul(delta_bu.try_reshape_like(&(batch, d_inner, d_state))?)?,
            )?;

        let y = ssm_state_cache
            .retaped::<T>()
            .try_matmul(c.try_reshape_like(&(batch, d_state, Const::<1>))?)?;
        let du = d.try_broadcast_like(&(batch, d_inner))?.try_mul(u)?;
        let y = y.try_reshape_like(&(batch, d_inner))?.try_add(du)?;

        Ok((y, ssm_state_cache))
    }
}
