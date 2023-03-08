/// This advanced example shows how to work with dfdx in a generic
/// training setting.
use dfdx::prelude::{tensor_collection::TensorCollection, *};

fn classification_train<
    Inp: Shape,
    Targ: Shape,
    Model: ModuleMut<Tensor<Inp, E, D, OwnedTape<E, D>>, Output = Tensor<Targ, E, D, OwnedTape<E, D>>>
        + TensorCollection<E, D>,
    E: Dtype,
    D: Device<E>,
    Opt: Optimizer<Model, D, E>,
    BatchIter: Iterator<Item = (Tensor<Inp, E, D>, Tensor<Targ, E, D>)>,
>(
    model: &mut Model,
    opt: &mut Opt,
    batches: BatchIter,
    batch_accum: usize,
) {
    let mut grads = model.alloc_grads();
    for (i, (inp, target)) in batches.enumerate() {
        let y = model.forward_mut(inp.traced_into(grads));
        let loss = cross_entropy_with_logits_loss(y, target);
        grads = loss.backward();
        if i % batch_accum == 0 {
            opt.update(model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }
    }
}

fn main() {}
