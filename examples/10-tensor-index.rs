//! Demonstrates how to select sub tensors (index) from tensors

use dfdx::{
    shapes::{Rank2, Rank3},
    tensor::{AsArray, Cpu, Tensor, TensorFromArray},
    tensor_ops::SelectTo,
};

fn main() {
    let dev: Cpu = Default::default();

    let a: Tensor<Rank3<4, 2, 3>, f32, Cpu> = dev.tensor([
        [[0.00, 0.01, 0.02], [0.10, 0.11, 0.12]],
        [[1.00, 1.01, 1.02], [1.10, 1.11, 1.12]],
        [[2.00, 2.01, 2.02], [2.10, 2.11, 2.12]],
        [[3.00, 3.01, 3.02], [3.10, 3.11, 3.12]],
    ]);

    // the easiest thing to do is to select a single element from axis 0.
    // we can specify the expected shape using the generic arguments of the select
    // method. Note that the axis 0 (4) has been removed in the shape
    let b = a.clone().select::<Rank2<2, 3>, _>(dev.tensor(0));
    assert_eq!(b.array(), a.array()[0]);

    // but we can also select multiple elements from axis 0!
    // note that our axis 0 is now 6 instead of 4,
    // *and* we have 6 indices in our index tensor.
    // the numebr of indices determines the size of the new tensor.
    let c = a
        .clone()
        .select::<Rank3<6, 2, 3>, _>(dev.tensor([0, 0, 1, 1, 2, 2]));
    dbg!(c.array());

    // a 1d array of indices in this case can also be used to
    // select from axis 1.
    // note here that we have the same number of indices
    // as the size of the axis 0 (4).
    let d = a.clone().select::<Rank2<4, 3>, _>(dev.tensor([0, 1, 0, 1]));
    dbg!(d.array());

    // of course we can also select multiple values from axis 1.
    // in this case we just specify multiple indices instead of a single one
    // note here that for `d`, we specified 4 indices. here we are specifying [[usize; 6]; 4]
    // indices, and we get a (4, 6, 3) tensor instead of a (4, 3) tensor
    let e = a.select::<Rank3<4, 6, 3>, _>(dev.tensor([
        [0; 6],
        [1; 6],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ]));
    dbg!(e.array());
}
