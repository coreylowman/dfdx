use crate::{
    arrays::{Dtype, Shape},
    devices::Device,
    gradients::{Merge, Tape},
    tensor::{make_tensor, Tensor},
};

pub(crate) trait UnaryKernel<Op, Inp: Shape, Out: Shape, E: Dtype>: Device {
    fn unary_fwd(
        &self,
        op: Op,
        inp: &Self::Storage<Inp, E>,
    ) -> Result<Self::Storage<Out, E>, Self::Err>;
    fn unary_bwd(
        &self,
        op: Op,
        inp: &Self::Storage<Inp, E>,
        grad_inp: &mut Self::Storage<Inp, E>,
        grad_out: &Self::Storage<Out, E>,
    ) -> Result<(), Self::Err>;
}

pub(crate) trait FullUnaryKernel<Op, Inp: Shape, Out: Shape, E: Dtype>: Device {
    fn unary_fwd(
        &self,
        op: Op,
        inp: &Self::Storage<Inp, E>,
    ) -> Result<Self::Storage<Out, E>, Self::Err>;
    fn unary_bwd(
        &self,
        op: Op,
        inp: &Self::Storage<Inp, E>,
        grad_inp: &mut Self::Storage<Inp, E>,
        out: &Self::Storage<Out, E>,
        grad_out: &Self::Storage<Out, E>,
    ) -> Result<(), Self::Err>;
}

pub(crate) trait BinaryKernel<Op, Lhs: Shape, Rhs: Shape, Out: Shape, E: Dtype>:
    Device
{
    fn binary_fwd(
        &self,
        op: Op,
        lhs: &Self::Storage<Lhs, E>,
        rhs: &Self::Storage<Rhs, E>,
    ) -> Result<Self::Storage<Out, E>, Self::Err>;

    fn binary_bwd(
        &self,
        op: Op,
        lhs: &Self::Storage<Lhs, E>,
        grad_lhs: &mut Self::Storage<Lhs, E>,
        rhs: &Self::Storage<Rhs, E>,
        grad_rhs: &mut Self::Storage<Rhs, E>,
        grad_out: &Self::Storage<Out, E>,
    ) -> Result<(), Self::Err>;
}

pub(crate) fn try_unary_op<
    Op: 'static + Clone,
    Inp: Shape,
    Out: Shape,
    E: Dtype,
    D: Device,
    T: Tape<D>,
>(
    op: Op,
    inp: Tensor<Inp, E, D, T>,
) -> Result<Tensor<Out, E, D, T>, D::Err>
where
    D: UnaryKernel<Op, Inp, Out, E>,
{
    let (inp, mut tape) = inp.split_tape();
    let storage = inp.device.unary_fwd(op.clone(), &inp.storage)?;
    let out = make_tensor(&inp.device, storage);
    let phantom_out = out.clone();
    tape.add_backward_op(move |grads| {
        let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
        inp.device.unary_bwd(op, &inp.storage, grad_inp, grad_out)?;
        Ok(())
    });
    Ok(out.put_tape(tape))
}

pub(crate) fn try_full_unary_op<
    Op: 'static + Copy,
    Inp: Shape,
    Out: Shape,
    E: Dtype,
    D: Device,
    T: Tape<D>,
>(
    op: Op,
    inp: Tensor<Inp, E, D, T>,
) -> Result<Tensor<Out, E, D, T>, D::Err>
where
    D: FullUnaryKernel<Op, Inp, Out, E>,
{
    let (inp, mut tape) = inp.split_tape();
    let storage = inp.device.unary_fwd(op, &inp.storage)?;
    let out = make_tensor(&inp.device, storage);
    let phantom_out = out.clone();
    tape.add_backward_op(move |grads| {
        let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
        inp.device
            .unary_bwd(op, &inp.storage, grad_inp, &phantom_out.storage, grad_out)?;
        Ok(())
    });
    Ok(out.put_tape(tape))
}

pub(crate) fn try_binary_op<
    Op: 'static + Copy,
    Lhs: Shape,
    Rhs: Shape,
    Out: Shape,
    E: Dtype,
    D: Device,
    LhsTape: Tape<D>,
    RhsTape: Tape<D>,
>(
    op: Op,
    lhs: Tensor<Lhs, E, D, LhsTape>,
    rhs: Tensor<Rhs, E, D, RhsTape>,
) -> Result<Tensor<Out, E, D, LhsTape>, D::Err>
where
    D: BinaryKernel<Op, Lhs, Rhs, Out, E>,
    LhsTape: Merge<RhsTape>,
{
    let (lhs, ltape) = lhs.split_tape();
    let (rhs, rtape) = rhs.split_tape();
    let mut tape = ltape.merge(rtape);
    let storage = lhs.device.binary_fwd(op, &lhs.storage, &rhs.storage)?;
    let out = make_tensor(&lhs.device, storage);
    let phantom_out = out.clone();
    tape.add_backward_op(move |grads| {
        let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out)?;
        lhs.device
            .binary_bwd(op, &lhs.storage, grad_lhs, &rhs.storage, grad_rhs, grad_out)?;
        Ok(())
    });
    Ok(out.put_tape(tape))
}
