/// This advanced example shows how to work with dfdx in a generic
/// training setting.
use dfdx::{optim::Sgd, prelude::*, tensor::AutoDevice};

/// Our generic training function. Works with any model/optimizer/loss function!
fn classification_train<
    // The input to our network, since we are training, we need it to implement Trace
    // so we can put gradients into it.
    Inp: Trace<E, D>,
    // The type of our label, we specify it here so we guaruntee that the dataset
    // and loss function both work on this type
    Lbl,
    // Our model just needs to implement these two things! ModuleMut for forward
    // and TensorCollection for optimizer/alloc_grads/zero_grads
    Model: ModuleMut<Inp::Traced, Error = D::Err> + TensorCollection<E, D>,
    // optimizer, pretty straight forward
    Opt: Optimizer<Model, D, E>,
    // our data will just be any iterator over these items. easy!
    Data: Iterator<Item = (Inp, Lbl)>,
    // Our loss function that takes the model's output & label and returns
    // the loss. again we can use a rust builtin
    Criterion: FnMut(Model::Output, Lbl) -> Loss,
    // the Loss needs to be able to call backward, and we also use
    // this generic as an output
    Loss: Backward<E, D, Err = D::Err> + AsArray<Array = E>,
    // Dtype & Device to tie everything together
    E: Dtype,
    D: Device<E>,
>(
    model: &mut Model,
    opt: &mut Opt,
    mut criterion: Criterion,
    data: Data,
    batch_accum: usize,
) -> Result<(), D::Err> {
    let mut grads = model.try_alloc_grads()?;
    for (i, (inp, lbl)) in data.enumerate() {
        let y = model.try_forward_mut(inp.traced(grads))?;
        let loss = criterion(y, lbl);
        let loss_value = loss.array();
        grads = loss.try_backward()?;
        if i % batch_accum == 0 {
            opt.update(model, &grads).unwrap();
            model.try_zero_grads(&mut grads)?;
        }
        println!("batch {i} | loss = {loss_value:?}");
    }
    Ok(())
}

fn main() {
    let dev = AutoDevice::default();

    type Model = Linear<10, 2>;
    type Dtype = f32;
    let mut model = dev.build_module::<Model, Dtype>();
    let mut opt = Sgd::new(&model, Default::default());

    // just some random data
    let mut data = Vec::new();
    for _ in 0..100 {
        let inp = dev.sample_normal::<Rank2<5, 10>>();
        let lbl = dev.tensor([[0.0, 1.0]; 5]);
        data.push((inp, lbl));
    }

    classification_train(
        &mut model,
        &mut opt,
        cross_entropy_with_logits_loss,
        data.into_iter(),
        1,
    )
    .unwrap();
}
