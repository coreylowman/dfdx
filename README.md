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

## Easy to use Optimizer API

See [examples/sgd.rs](examples/sgd.rs) and [examples/regression.rs](examples/regression.rs) for more examples.

```rust

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // input & target data
    let mut x = Tensor2D::<64, 5>::rand(&mut rng);
    let mut y = Tensor2D::<64, 2>::rand(&mut rng);

    // construct optimizer
    let mut opt: Sgd<Linear<5, 2>> = Default::default();

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

### Batching

Batching is currently implemented with this trait:

```rust
// src/tensor/tensor.rs
pub trait Batch {
    type Batched<const B: usize>: Tensor;
}

// src/tensor/tensor_impl.rs
impl Batch for Tensor0D {
    type Batched<const B: usize> = Tensor1D<B>;
}

impl<const N: usize> Batch for Tensor1D<N> {
    type Batched<const B: usize> = Tensor2D<B, N>;
}
```

This is where the nightly dependency comes in!

The Module API requires inputs & outputs to implement the Batch trait, which is why all the examples have a batch dimension.

### Module & ModuleChain

I like the definition for the Module trait a lot:

```rust
pub trait Module: Init + Taped + Default {
    type Input: Tensor + Batch;
    type Output: Tensor + Batch;

    fn forward<const B: usize>(
        &mut self,
        input: &mut <Self::Input as Batch>::Batched<B>,
    ) -> <Self::Output as Batch>::Batched<B>;
}
```

It just takes its input and produces an output!

What's cool is how we can use this to implement a ModuleChain (chaining two modules together):

```rust
#[derive(Default, Debug)]
pub struct ModuleChain<M1: Module, M2: Module<Input = M1::Output>> {
    first: M1,
    second: M2,
}
```

We can require the second module's input to be the same as the first module's output! SO COOL!

(Of course ModuleChain also implements Module!)

### Optimizer has all methods of underlying module for free!

Yay [DerefMut](https://doc.rust-lang.org/std/ops/trait.DerefMut.html)!

