use crate::{
    gradients::{Merge, Tape},
    shapes::{Dtype, Shape},
    tensor::{DeviceStorage, PutTape, SplitTape, Tensor},
};

pub trait UnaryKernel<Op, E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &Self::Storage<S, E>,
        grad_inp: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err>;
}

pub trait BinaryKernel<Op, E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err>;

    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        grad_lhs: &mut Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
        grad_rhs: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err>;
}

pub(crate) fn try_unary_op<
    Op: 'static + Clone,
    S: Shape,
    E: Dtype,
    D: UnaryKernel<Op, E>,
    T: Tape<D>,
>(
    op: Op,
    inp: Tensor<S, E, D, T>,
) -> Result<Tensor<S, E, D, T>, D::Err> {
    let (inp, mut tape) = inp.split_tape();
    let storage = inp.device.forward(op.clone(), &inp.storage)?;
    let out = inp.device.upgrade(storage);
    let phantom_out = out.clone();
    tape.try_alloc_grad(&inp)?;
    tape.try_alloc_grad(&out)?;
    tape.add_backward_op(move |grads| {
        let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
        inp.device.backward(op, &inp.storage, grad_inp, grad_out)?;
        Ok(())
    });
    Ok(out.put_tape(tape))
}

pub(crate) fn try_binary_op<
    Op: 'static + Copy,
    S: Shape,
    E: Dtype,
    D: BinaryKernel<Op, E>,
    RhsTape: Tape<D>,
    LhsTape: Tape<D> + Merge<RhsTape>,
>(
    op: Op,
    lhs: Tensor<S, E, D, LhsTape>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Result<Tensor<S, E, D, LhsTape>, D::Err> {
    let (lhs, ltape) = lhs.split_tape();
    let (rhs, rtape) = rhs.split_tape();
    let mut tape = ltape.merge(rtape);
    let storage = lhs.device.forward(op, &lhs.storage, &rhs.storage)?;
    let out = lhs.device.upgrade(storage);
    let phantom_out = out.clone();
    tape.try_alloc_grad(&lhs)?;
    tape.try_alloc_grad(&rhs)?;
    tape.try_alloc_grad(&out)?;
    tape.add_backward_op(move |grads| {
        let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out);
        lhs.device
            .backward(op, &lhs.storage, grad_lhs, &rhs.storage, grad_rhs, grad_out)?;
        Ok(())
    });
    Ok(out.put_tape(tape))
}
