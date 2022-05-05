# STAG: Strongly Typed Auto Grad

Reverse Mode Auto Differentiation[1] in Rust.

```rust
// Declare the structure of our feedforward model - each module is executed sequentially
type MLP = (
    (Linear<10, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize input & output data
    let x: Tensor2D<64, 10> = Tensor2D::randn(&mut rng);
    let y: Tensor2D<64, 2> = Tensor2D::randn(&mut rng);

    // initialize our MLP with uniform random weights
    let mut module: MLP = Default::default();
    module.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    // trace x through the module
    let x = x.trace();
    let pred = module.forward(&x);
    let loss = (&y - pred).square().mean();
    
    // compute gradients
    let mut sgd = Sgd { lr: 1e-2 };
    let (_, gradients) = sgd.compute_gradients(loss);

    // update the MLP
    module.update_with_tape(&gradients);
}
```

[1] https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation


## Features

1. 👌 Simple Neural Networks API, completely type checked at compile time. See [examples/regression.rs](examples/regression.rs)
2. 📈 Easy to use Optimizer API
3. 💡 Tensor sizes & operations type checked at compile time
4. ✔ No unsafe rust code

## Fun/notable implementation details

### Module

```rust
pub trait Module<Input>: Default + UpdateWithTape {
    type Output;
    fn forward(&self, input: &Input) -> Self::Output;
}
```

The Module trait is simple yet powerful! The generic type variable for the input allows a single struct to implement module for multiple inputs. *This is how batching is implented!* The associated type variable for output enables a number of other nice features, but also allows the implementation to control what the output is.

### Feedforward (a.k.a Sequential) modules & Tuples

The associated Output type makes combining multiple modules to be executed sequentially very easy:

```rust
impl<Input, A, B> Module<Input> for (A, B)
where
    Input: Tensor,
    A: Module<Input>,        // A is a module that takes Input
    B: Module<A::Output>,    // B is a module that takes A's Output
{
    type Output = B::Output; // the output of this is B's Output
    fn forward(&self, x: &Input) -> Self::Output {
        let x = self.0.forward(x);
        self.1.forward(&x)
    }
}
```

What's also nice is that we can use tuple's as the container instead of introducing a whole new struct. *This is not possible in other languages!*

### Type checked backward

tl;dr: If you forget to include a call to `trace()`, the program won't compile!

```diff
+let x = x.trace();
let pred = module.forward(x);
let loss = (&y - pred).square().mean();
let (loss_v, gradients) = sgd.compute_gradients(loss);
```

In order to compute the gradients from a computation graph, the actual gradient tape is needed.
In STAG, the gradient tape is transferred to the output of operations. This means the loss tensor
should have a tape with it, since it is the final output of the whole computation graph.

STAG requires this via various function definitions:

1. The `Optimizer` trait requires a 0D tensor with a `WithTape` type parameter. The `WithTape` struct is a `TapeHolder` and has the actual `GradientTape`.

```rust
pub trait Optimizer {
    fn compute_gradients(&mut self, loss: Tensor0D<WithTape>) -> (f32, Box<GradientTape>);
}
```

2. The `backward` function accepts a generic tensor, also with the `TapeHolder` generic set to `WithTape`!

```rust
fn backward<T: Tensor<TapeHolder = WithTape>>(t: T) -> Box<crate::gradients::GradientTape> {
    ...
}
```

### Gradient tape is not reference counted!

Since all operations result in exactly 1 child, we can always move the gradient tape to the child of the last operation. This means we know exactly which tensor owns the gradient tape!
