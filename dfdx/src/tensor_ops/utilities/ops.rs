use crate::{
    shapes::{Dtype, HasShape, Shape},
    tensor::{Merge, PutTape, SplitTape, Storage, Tape, Tensor, Tensorlike},
};
use std::borrow::Cow;

pub trait UnaryKernel<Op, E: Dtype>: Storage<E> {
    const BACKWARD_WITHOUT_INP: bool;
    const BACKWARD_WITHOUT_DATA: bool;
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err>;
}

pub trait BinaryKernel<Op, E: Dtype>: Storage<E> {
    const BACKWARD_WITHOUT_DATA: bool;
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &impl Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
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
    let inp_ghost = inp.ghost();
    let dev = inp.device.clone();
    if !T::OWNS_TAPE || D::BACKWARD_WITHOUT_DATA {
        let out = inp_ghost.dev.forward(op.clone(), Cow::Owned(inp))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, &inp_ghost, grad_inp, &out_ghost, grad_out)
        });
        Ok(out.put_tape(tape))
    } else if D::BACKWARD_WITHOUT_INP {
        let out = inp_ghost.dev.forward(op.clone(), Cow::Owned(inp))?;
        let out_ghost = out.ghost();
        let out_clone = out.clone();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, &inp_ghost, grad_inp, &out_clone, grad_out)
        });
        Ok(out.put_tape(tape))
    } else {
        let out = inp.device.forward(op.clone(), Cow::Borrowed(&inp))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, &inp, grad_inp, &out_ghost, grad_out)
        });
        Ok(out.put_tape(tape))
    }
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
    let lhs_ghost = lhs.ghost();
    let rhs_ghost = rhs.ghost();
    let mut tape = ltape.merge(rtape);
    if !LhsTape::OWNS_TAPE || D::BACKWARD_WITHOUT_DATA {
        let out = lhs_ghost
            .dev
            .forward(op, Cow::Owned(lhs), Cow::Owned(rhs))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs_ghost
                .dev
                .backward(op, &lhs_ghost, grad_lhs, &rhs_ghost, grad_rhs, grad_out)
        });
        Ok(out.put_tape(tape))
    } else {
        let out = lhs
            .device
            .forward(op, Cow::Borrowed(&lhs), Cow::Borrowed(&rhs))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs.device
                .backward(op, &lhs, grad_lhs, &rhs, grad_rhs, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}
