use crate::{
    shapes::{Dtype, HasShape, Shape},
    tensor::{DeviceStorage, Merge, PutTape, SplitTape, Tape, Tensor},
};

pub trait UnaryKernel<Op, E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

pub trait BinaryKernel<Op, E: Dtype>: DeviceStorage {
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;

    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &Tensor<S, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<S, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

pub(crate) fn try_unary_op<
    Op: 'static + Clone,
    S: Shape,
    E: Dtype,
    D: UnaryKernel<Op, E>,
    T: Tape<E, D>,
>(
    op: Op,
    inp: Tensor<S, E, D, T>,
) -> Result<Tensor<S, E, D, T>, D::Err> {
    let (inp, mut tape) = inp.split_tape();
    let out = inp.device.forward(op.clone(), &inp)?;
    let phantom_out = out.clone();
    tape.try_alloc_grad(&inp)?;
    tape.try_alloc_grad(&out)?;
    tape.add_backward_op(move |grads| {
        let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
        inp.device.backward(op, &inp, grad_inp, grad_out)?;
        Ok(())
    });
    Ok(out.put_tape(tape))
}

pub(crate) fn try_binary_op<
    Op: 'static + Copy,
    S: Shape,
    E: Dtype,
    D: BinaryKernel<Op, E>,
    RhsTape,
    LhsTape: Tape<E, D> + Merge<RhsTape>,
>(
    op: Op,
    lhs: Tensor<S, E, D, LhsTape>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Result<Tensor<S, E, D, LhsTape>, D::Err> {
    assert_eq!(lhs.shape(), rhs.shape());
    let (lhs, ltape) = lhs.split_tape();
    let (rhs, rtape) = rhs.split_tape();
    let mut tape = ltape.merge(rtape);
    let out = lhs.device.forward(op, &lhs, &rhs)?;
    let phantom_out = out.clone();
    tape.try_alloc_grad(&lhs)?;
    tape.try_alloc_grad(&rhs)?;
    tape.try_alloc_grad(&out)?;
    tape.add_backward_op(move |grads| {
        let (grad_lhs, grad_rhs, grad_out) = grads.muts_and_ref(&lhs, &rhs, &phantom_out);
        lhs.device
            .backward(op, &lhs, grad_lhs, &rhs, grad_rhs, grad_out)?;
        Ok(())
    });
    Ok(out.put_tape(tape))
}
