//! Demonstrates broadcasting tensors to different sizes, and axis reductions
//! with BroadcastTo and ReduceTo

use dfdx::arrays::Axis;
use dfdx::tensor::{tensor, HasArrayData, Tensor1D, Tensor2D, Tensor4D};
use dfdx::tensor_ops::BroadcastTo;

fn main() {
    let a: Tensor1D<3> = tensor([1.0, 2.0, 3.0]);

    // to broadcast, use `BroadcastTo::broadcast()` and specify
    // the output type. the axes that are broadcast are inferred for you!
    let b: Tensor2D<5, 3> = a.broadcast();
    assert_eq!(b.data(), &[[1.0, 2.0, 3.0]; 5]);

    // we can really broadcast any axes on either side
    // here a (5,3) tensor is broacast to (7,5,3,2).
    // so 7 is added in front, and 2 is added last
    let c: Tensor4D<7, 5, 3, 2> = b.broadcast();
    assert_eq!(c.data(), &[[[[1.0; 2], [2.0; 2], [3.0; 2]]; 5]; 7]);

    // the opposite of broadcast is reducing
    // we've already introduced one reduction which is mean
    let d: Tensor2D<5, 3> = c.mean();
    assert_eq!(d.data(), &[[1.0, 2.0, 3.0]; 5]);

    // generally you can just specify the output type
    // and the reduction & broadcast will work.
    // sometimes it's ambiguous though
    let e: Tensor1D<1> = tensor([1.0]);

    // here rust doesn't know if the new axis is the first or second
    // so we have to explicitly tell it
    let f: Tensor2D<1, 1> = BroadcastTo::<_, Axis<1>>::broadcast(e);

    // reductions have the same problem when it's ambiguous
    let _: Tensor1D<1> = f.mean::<_, Axis<0>>();
}
