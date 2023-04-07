use crate::{
    shapes::{Dtype, HasShape, Shape},
    tensor::{DeviceStorage, GhostTensor, Merge, PutTape, SplitTape, Tape, Tensor},
};

pub trait UnaryKernel<Op, E: Dtype>: DeviceStorage {
    const BACKWARD_WITHOUT_INP: bool;
    const BACKWARD_WITHOUT_DATA: bool;
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: Result<&Tensor<S, E, Self>, Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: Result<&Tensor<S, E, Self>, &GhostTensor<S, E, Self>>,
        grad_inp: &mut Self::Vec<E>,
        out: Result<&Tensor<S, E, Self>, &GhostTensor<S, E, Self>>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

pub trait BinaryKernel<Op, E: Dtype>: DeviceStorage {
    const BACKWARD_WITHOUT_DATA: bool;
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: Result<&Tensor<S, E, Self>, Tensor<S, E, Self>>,
        rhs: Result<&Tensor<S, E, Self>, Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err>;
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: Result<&Tensor<S, E, Self>, &GhostTensor<S, E, Self>>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: Result<&Tensor<S, E, Self>, &GhostTensor<S, E, Self>>,
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
    let inp_ghost = inp.ghost();
    let dev = inp.device.clone();
    if !T::OWNS_TAPE || D::BACKWARD_WITHOUT_DATA {
        let out = inp_ghost.dev.forward(op.clone(), Err(inp))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, Err(&inp_ghost), grad_inp, Err(&out_ghost), grad_out)
        });
        Ok(out.put_tape(tape))
    } else if D::BACKWARD_WITHOUT_INP {
        let out = inp_ghost.dev.forward(op.clone(), Err(inp))?;
        let out_ghost = out.ghost();
        let out_clone = out.clone();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, Err(&inp_ghost), grad_inp, Ok(&out_clone), grad_out)
        });
        Ok(out.put_tape(tape))
    } else {
        let out = inp.device.forward(op.clone(), Ok(&inp))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            dev.backward(op, Ok(&inp), grad_inp, Err(&out_ghost), grad_out)
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
        let out = lhs_ghost.dev.forward(op, Err(lhs), Err(rhs))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs_ghost.dev.backward(
                op,
                Err(&lhs_ghost),
                grad_lhs,
                Err(&rhs_ghost),
                grad_rhs,
                grad_out,
            )
        });
        Ok(out.put_tape(tape))
    } else {
        let out = lhs.device.forward(op, Ok(&lhs), Ok(&rhs))?;
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs.device
                .backward(op, Ok(&lhs), grad_lhs, Ok(&rhs), grad_rhs, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}
