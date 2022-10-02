//! Demonstrates how to select sub tensors (index) from tensors

use dfdx::tensor::{tensor, HasArrayData, Tensor2D, Tensor3D};
use dfdx::tensor_ops::Select1;

fn main() {
    let a: Tensor3D<3, 2, 3> = tensor([
        [[0.00, 0.01, 0.02], [0.10, 0.11, 0.12]],
        [[1.00, 1.01, 1.02], [1.10, 1.11, 1.12]],
        [[2.00, 2.01, 2.02], [2.10, 2.11, 2.12]],
    ]);

    // the easiest thing to do is to select a single element from axis 0
    let b: Tensor2D<2, 3> = a.clone().select(&0);
    assert_eq!(b.data(), &a.data()[0]);

    // but we can also select multiple elements from axis 0!
    let _: Tensor3D<6, 2, 3> = a.clone().select(&[0, 0, 1, 1, 2, 2]);

    // a 1d array of indices in this case can also mean
    // select from the second axis. this is determined by two things:
    // 1. we have 3 usize's in our indices, and 3 is the size of the first dimension
    // 2. the output type has lost the middle axis, which means the usizes are reducing that axis
    let _: Tensor2D<3, 3> = a.clone().select(&[0, 1, 0]);

    // of course we can also select multiple values from the first axis also.
    // in this case we just specify multiple indices instead of a single one
    let _: Tensor3D<3, 4, 3> = a.select(&[[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]]);
}
