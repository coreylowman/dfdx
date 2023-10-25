use dfdx::prelude::*;

fn main() {
    let dev = AutoDevice::default();

    // Let's build our model similar to the first example.
    type Model = LinearConstConfig<2, 5>;
    let mut model = dev.build_module::<f32>(Model::default());

    // Let's sample two different tensor shapes to pass to forward.
    let unbatched_x: Tensor<Rank1<2>, f32, _> = dev.sample_uniform();
    let batched_x: Tensor<Rank2<10, 2>, f32, _> = dev.sample_uniform();

    // The Module trait lets us call either forward or forward_mut on a module.
    use dfdx_nn::Module;

    // Here we call forward on the unbatched tensor.
    let unbatched_y = model.forward(unbatched_x);
    assert_eq!(unbatched_y.shape(), &(Const::<5>,));

    // And we can use the same model & the same method to forward batched data as well.
    // NOTE: here we are calling forward_mut, which may mutate the model.
    let batched_y = model.forward_mut(batched_x);
    assert_eq!(batched_y.shape(), &(Const::<10>, Const::<5>));
}
