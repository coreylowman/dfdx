# STAG: Strongly Typed Auto Grad

Reverse Mode Auto Differentiation[1] in Rust.

NOTE: Depends on nightly rust for const generic associated types.

[1] https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation

## Compile time tensor sizes & operations

```
use stag::nn::{Linear, ModuleChain, ReLU, Tanh};
use stag::prelude::*;

type MyMLP = chain_modules!(Linear<10, 32>, ReLU<Tensor1D<32>>, Linear<32, 32>, ReLU<Tensor1D<32>>, Linear<32, 2>, Tanh<Tensor1D<2>>);

fn main() {
    // construct the MLP as defined above with all parameters filled with 0s
    let mut model: MyMLP = Default::default();

    // create a 1x10 tensor filled with 0s
    let mut x: Tensor2D<1, 10> = Default::default();

    // pass through the MLP
    let y = model.forward(&mut x);

    println!("{:#}", y.data());
    // [[0, 0]]
}
```

