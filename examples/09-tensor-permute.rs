//! Demonstrates how to re-order (permute/transpose) the axes of a tensor

use dfdx::arrays::Axes3;
use dfdx::tensor::{Tensor3D, TensorCreator};
use dfdx::tensor_ops::PermuteTo;

fn main() {
    let a: Tensor3D<3, 5, 7> = TensorCreator::zeros();

    // permuting is as easy as just expressing the desired type
    let b: Tensor3D<7, 5, 3> = a.permute();

    // we can do any of the expected combinations!
    let _: Tensor3D<5, 7, 3> = b.permute();

    // just like broadcast/reduce there are times when
    // inference is impossible because of ambiguities
    let c: Tensor3D<1, 1, 1> = TensorCreator::zeros();

    // when axes have the same sizes you'll have to indicate
    // the axes explicitly to get around this
    let _: Tensor3D<1, 1, 1> = PermuteTo::<_, Axes3<1, 0, 2>>::permute(c);
}
