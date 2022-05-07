# stag: Strongly Typed Auto Grad

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

    // initialize our MLP with all zero weights
    let mut mlp: MLP = Default::default();

    // randomize the weights according to a uniform random distribution
    mlp.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    // execute the MLP - x.trace() tells stag to collect gradient information
    let pred = mlp.forward(x.trace());
    let loss = (&y - pred).square().mean();

    // run backprop to get the gradients
    let gradients = loss.backward();
    
    // update the MLP with SGD!
    let mut sgd = Sgd::new(1e-2);
    sgd.update(&mut mlp, gradients);
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
pub trait Module<Input>: Default + CanUpdateWithGrads {
    type Output;
    fn forward(&self, input: &Input) -> Self::Output;
}
```

The Module trait is simple yet powerful! The generic type variable for the input allows a single struct to implement module for multiple inputs. *This is how batching is implented!* The associated type variable for output enables a number of other nice features, but also allows the implementation to control what the output is.

### Tuples represent feedforward (a.k.a sequential) modules

Since we can implement traits for tuples, which is *not possible in other languages* AFAIK, they provide a very nice frontend
for sequentially executing modules.

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

We've implemented Module for Tuples up to 6 elements, but *you can arbitrarily nest them*!

### Type checked backward

tl;dr: If you forget to include a call to `trace()`, the program won't compile!

```diff
-let pred = module.forward(x);
+let pred = module.forward(x.trace());
let loss = (&y - pred).square().mean();
let gradients = loss.backward();
```

In order to compute the gradients from a computation graph, the actual gradient tape is needed.
In `stag`, the gradient tape is transferred to the output of operations. This means the loss tensor
should have a tape with it, since it is the final output of the whole computation graph.

`stag` requires this at compile time via the `backward` function.
It accepts a generic tensor, but with the `TapeHolder` generic set to `WithTape`!

```rust
pub fn backward<T: Tensor<TapeHolder = WithTape>>(t: T) -> Box<crate::gradients::GradientTape> {
    ...
}
```

### Gradient tape is not reference counted!

Since all operations result in exactly 1 child, we can always move the gradient tape to the child of the last operation. This means we know exactly which tensor owns the gradient tape!

One advanced use case requires that tensors be re-used multiple times in a computation graph.
This can be handled by duplicating the tensor, and manually moving the gradient tape around.
See [examples/multi_head.rs](examples/multi_head.rs) for an example.

### Array Backend

Our [src/array_ops](src/array_ops/) backend for computing results operations
is built using __recursive trait definitions__.

The main idea behind this is similar to recursion or induction proofs. First we specify
the base trait, and then we specify the recursive trait.

A simple example is counting the number of elements in an arbitrarily nested array
at compile time.

First we specify the trait we want to do this:

```rust
pub trait CountElements {
    const NUM_ELEMENTS: usize;
}
```

Now for the base case (assuming these will be arrays of floats), is just a single floating point number:

```rust
impl CountElements for f32 {
    const NUM_ELEMENTS: usize = 1;
}
```

And finally the recursive trait:

```rust
impl<T: CountElements, const M: usize> CountElements for [T; M] {
    const NUM_ELEMENTS: usize = M * T::NUM_ELEMENTS;
}
```

Notice the restriction on T also implementing `CountElements`. This allows us to use `T::NUM_ELEMENTS` in the trait body.

Another few powerful things recursive traits can do:
1. Map all elements of arbitarily nested arrays using a function
2. Add two arrays together
3. Reduce an array to one number
4. Even more!

Encourage you to check out all the code in [src/array_ops](src/array_ops/)!

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
