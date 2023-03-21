//! Demonstrates how to select or gather sub tensors (index) from tensors

use dfdx::{
    shapes::Rank3,
    tensor::{AsArray, AutoDevice, Tensor, TensorFrom},
    tensor_ops::{GatherTo, SelectTo},
};

fn main() {
    let dev = AutoDevice::default();

    let a: Tensor<Rank3<4, 2, 3>, f32, _> = dev.tensor([
        [[0.00, 0.01, 0.02], [0.10, 0.11, 0.12]],
        [[1.00, 1.01, 1.02], [1.10, 1.11, 1.12]],
        [[2.00, 2.01, 2.02], [2.10, 2.11, 2.12]],
        [[3.00, 3.01, 3.02], [3.10, 3.11, 3.12]],
    ]);

    // the easiest thing to do is to `select` a single value from a given axis.
    // to do that, you need indices with a shape up to the axis you are select from.
    // for example, given shape (M, N, O), here are the index shapes for each axis:
    // - Axis 0: index shape ()
    // - Axis 1: index shape (M, )
    // - Axis 2: index shape (M, N)
    // here we select from axis 0 so we just need 1 value.
    let b = a.clone().select(dev.tensor(0));
    assert_eq!(b.array(), a.array()[0]);

    // to `select` from axis 1, we use a tensor with shape (4,)
    let d = a.clone().select(dev.tensor([0, 1, 0, 1]));
    assert_eq!(
        d.array(),
        [
            [0.00, 0.01, 0.02],
            [1.10, 1.11, 1.12],
            [2.00, 2.01, 2.02],
            [3.10, 3.11, 3.12]
        ]
    );

    // We can also `gather` multiple elements from each axis. This lets you grab
    // the same elements multiple times! This requires an index with shape similar to select,
    // but with an extra dimension at the end that says how many elements to gather from the axis:
    // - Axis 0: index shape (Z, )
    // - Axis 1: index shape (M, Z)
    // - Axis 2: index shape (M, N, Z)
    // here, we `gather` from axis 0 because we have a 1d tensor. the new size will be (6, 2, 3)!
    let c = a.clone().gather(dev.tensor([0, 0, 1, 1, 2, 2]));
    dbg!(c.array());

    // and similarly, we can `gather` from axis 1 with a 2d tensor. the new size will be (4, 6, 3)!
    let e = a.gather(dev.tensor([[0; 6], [1; 6], [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]));
    dbg!(e.array());
}
