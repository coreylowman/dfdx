//! Demonstrates how to build a neural network that has multiple
//! outputs using `SplitInto`.

use dfdx::nn::{Linear, Module, SplitInto};
use dfdx::tensor::{tensor, Tensor1D};

fn main() {
    // SplitInto accepts a tuple of modules. Each one of the items in the
    // tuple must accept the same type of input.
    // Note that here, both of the linears have the same size input (1)
    let m: SplitInto<(Linear<1, 3>, Linear<1, 5>)> = Default::default();

    // when we forward data through, we get a tuple back!
    let x = tensor([1.0]);
    let _: (Tensor1D<3>, Tensor1D<5>) = m.forward(x);
}
