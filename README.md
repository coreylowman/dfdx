# STAG: Strongly Typed Auto Grad

Reverse Mode Auto Differentiation[1] in Rust.

NOTE: Depends on nightly rust for const generic associated types.

[1] https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation

## Tensor sizes & operations checked at compile time

See [examples/tensor.rs](examples/tensor.rs) for more tensor operation examples.

```rust
use stag::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // 3x3 matrix filled with 0s
    let mut x: Tensor2D<3, 3> = Default::default();

    // fill matrix with random data drawn from Standard distribution
    x.randomize(&mut rng, &Standard);
    println!("x={:#}", x.data());
    // x=[[0.80145925, 0.7311134, 0.55528885],
    // [0.77346015, 0.809342, 0.025844634],
    // [0.6714777, 0.58415926, 0.87062806]]
}
```

## Neural networks checked at compile time.

See [examples/linear.rs](examples/linear.rs), [examples/chain.rs](examples/chain.rs), and [examples/regression.rs](examples/regression.rs) for more examples.

```rust
use stag::nn::{Linear, ModuleChain, ReLU, Tanh};
use stag::prelude::*;

// NOTE: this is just syntactic sugar for combining multiple ModuleChain structs together.
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

