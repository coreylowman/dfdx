# STAG: Strongly Typed Auto Grad

Reverse Mode Auto Differentiation[1] in Rust.

[1] https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation


## 👌 Simple Neural Networks API, completely type checked at compile time.

See [examples/linear.rs](examples/linear.rs), [examples/chain.rs](examples/chain.rs), and [examples/regression.rs](examples/regression.rs) for more examples.

```rust
use stag::nn::{Linear, ReLU, Tanh};
use stag::prelude::*;

// tuple's represent sequential or chained modules
type MLP = (
    (Linear<16, 8>, ReLU),
    (Linear<8, 4>, ReLU),
    (Linear<4, 2>, Tanh),
);

fn main() {
    // construct a multi layer MLP with ReLU activations and all weights/biases filled with 0s
    let mut model: MLP = Default::default();

    // create a 1x10 tensor filled with 0s
    let mut x: Tensor1D<16> = Default::default();

    // pass through the MLP
    let y = model.forward(&mut x);

    println!("{:#}", y.data());
    // [0, 0]
}
```

## 💡 Tensor sizes & operations type checked at compile time

See [examples/tensor.rs](examples/tensor.rs) for more tensor operation examples.

```rust
use stag::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // 3x3 matrix filled with 0s
    let mut x: Tensor2D<3, 3> = Tensor2D::default();

    // fill matrix with random data drawn from Standard distribution
    x.randomize(&mut rng, &Standard);
    println!("x={:#}", x.data());
    // x=[[0.80145925, 0.7311134, 0.55528885],
    // [0.77346015, 0.809342, 0.025844634],
    // [0.6714777, 0.58415926, 0.87062806]]
}
```

## 📄 Batching completely supported by type system

Since the `Module` trait is generic, we can implement module's for multiple inputs/outputs.
To support batching all we have to do is impl Module a second time with a batch dimension
added to inputs/outputs!

See [src/nn/linear.rs](src/nn/linear.rs) for an example implementation.

NOTE: Unfortunately because of the ModuleChain currently works, a model constructed
using ModuleChain can't call forward with two different data types.

```rust
use stag::nn::Linear;
use stag::prelude::*;

fn main() {
    let mut model = Linear::<10, 5>::default();

    // create a 1x10 tensor filled with 0s
    let mut a: Tensor1D<10> = Tensor1D::default();

    // create a 64x10 tensor filled with 0s
    let mut b: Tensor2D<64, 10> = Tensor2D::default();

    // yay both of these work!
    let y = model.forward(&mut a);
    let z = model.forward(&mut b);
}
```

## 📈 Easy to use Optimizer API

See [examples/sgd.rs](examples/sgd.rs) and [examples/regression.rs](examples/regression.rs) for more examples.

```rust

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // input & target data
    let mut x = Tensor2D::<64, 5>::rand(&mut rng);
    let mut y = Tensor2D::<64, 2>::rand(&mut rng);

    // construct optimizer
    let mut opt = Sgd::new(SgdConfig::default(), Linear::<5, 2>::default());

    // initialize weights of underlying module
    opt.init(&mut rng);

    // forward input through network while tracking derivatives
    let mut output = opt.forward_with_derivatives(&mut x);

    // compute loss
    let mut loss = (&mut output - &mut y).square().mean();

    // run backprop & apply gradients
    opt.step(&mut loss);
}
```

## Interesting implementation details

### No Arc/RefCell (& why tensors require mutability)

Since all operations in a computation graph have exactly 1 child, we can always move the gradient tape to the child of the last operation. This means we know exactly which tensor owns the gradient tape!

This is also why we have to mark all the tensors as mut and pass them around with &mut to the operations. Every operation could potentially pull the gradient tape out of the tensor!.

### Module & ModuleChain

I'm partial to the Module trait:

```rust
pub trait Module<I>: Default {
    type Output;
    fn forward(&mut self, input: &mut I) -> Self::Output;
}
```
This is nice because we can impl Module for different inputs for the same struct, which is how batching is implemented!

This also enables an easy sequential/chaining functionality with tuples, by implementing Module for a tuple of modules:

```rust
impl<Input, A, B> Module<Input> for (A, B)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
{
    type Output = B::Output;
    fn forward(&mut self, x: &mut INPUT) -> Self::Output {
        self.1.forward(&mut self.0.forward(x))
    }
}
```

### Optimizer has all methods of underlying module for free!

Yay [DerefMut](https://doc.rust-lang.org/std/ops/trait.DerefMut.html)!

