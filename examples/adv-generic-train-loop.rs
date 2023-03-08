/// This advanced example shows how to work with dfdx in a generic
/// training setting.
use dfdx::prelude::{tensor_collection::TensorCollection, *};

fn classification_train<
    Inp: Trace<E, D>,
    Lbl,
    Model: ModuleMut<Inp::Traced, Error = D::Err> + TensorCollection<E, D>,
    Opt: Optimizer<Model, D, E>,
    BatchIter: Iterator<Item = (Inp, Lbl)>,
    Loss: Backward<E, D, Err = D::Err>,
    LossFn: FnMut(Model::Output, Lbl) -> Loss,
    E: Dtype,
    D: Device<E>,
>(
    model: &mut Model,
    opt: &mut Opt,
    mut loss_fn: LossFn,
    batches: BatchIter,
    batch_accum: usize,
) -> Result<(), D::Err> {
    let mut grads = model.try_alloc_grads()?;
    for (i, (inp, lbl)) in batches.enumerate() {
        let y = model.try_forward_mut(inp.traced_into(grads))?;
        let loss = loss_fn(y, lbl);
        grads = loss.try_backward()?;
        if i % batch_accum == 0 {
            opt.update(model, &grads).unwrap();
            model.try_zero_grads(&mut grads)?;
        }
    }
    Ok(())
}

fn main() {}
