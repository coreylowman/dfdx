# dfdx: strongly typed deep learning in Rust

Ergonomics & safety focused deep learning in Rust. Main features include:

1. Tensor library, complete with const generic shapes, activation functions, and more.
2. Safe & Easy to use neural network building blocks.
3. Standard deep learning optimizers such as Sgd and Adam.
4. Reverse mode auto differentiation[1] implementation.

`dfdx` is on [crates.io](https://crates.io/crates/dfdx)! Use by adding this to your `Cargo.toml`:

```toml
dfdx = "0.6.0"
```

See the documentation at [docs.rs/dfdx](https://docs.rs/dfdx).

[1] https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation

## Design Goals

1. Easy to use frontend interface.
2. Easy to understand/maintain internals. Keep levels of indirection to a minimum.
3. As much at compile time as possible (i.e. don't compile if something is not correct).
4. Maximize performance.
5. Keep internals as flexible as possible.
6. No unsafe code[1]
7. No Rc/RefCells[2]

[1] Currently the only unsafe calls are for matrix multiplication, and instantiating large arrays directly on the heap.

[2] There is only 1 usage of RefCell in the `nn::Dropout` layer to make it's underlying rng easy to use.

## Features

1. 👌 Simple Neural Networks API, completely type checked at compile time. See [examples/regression.rs](examples/regression.rs)

```rust
type MLP = (
    (Linear<10, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() {
    let mlp: MLP = Default::default();
    let x: Tensor1D<10> = Tensor1D::zeros();
    let y /*: Tensor1D<2>*/ = mlp.forward(x);
    println!("{:?}", y);
    mlp.save("checkpoint.npz")?;
}
```

2. 📈 Ergnomic & safe Optimizer API

```rust
let mut model = ...
let mut sgd = Sgd::new(1e-2, Some(Momentum::Nesterov(0.9)));

let loss: Tensor0D<OwnedTape> = ...

// run backprop to get the gradients
let gradients = loss.backward();
sgd.update(&mut model, gradients);
```

3. Tensors are backed by normal rust arrays, making it easy to access the underlying data!
```rust
let t0: Tensor0D = Tensor0D::zeros();
assert_eq!(t0.data(), &0.0);

let t1 /*: Tensor1D<3>*/ = Tensor1D::new([1.0, 2.0, 3.0]);
assert_eq!(t1.data(), &[1.0, 2.0, 3.0]);

let t2: Tensor2D<2, 3> = Tensor2D::ones();
assert_eq!(t2.data(), &[[1.0; 3]; 2]);
```

4. 💡 Tensor sizes, operations, gradient computations all type checked at compile time
5. 💪 Full power of rust compiler & llvm optimizations (because all shapes of arrays are known at compile time!)
6. Minimal runtime costs - there are no Rc/Refcells used in this implementation!

## Fun/notable implementation details

### Module

```rust
pub trait Module<Input>: Default + CanUpdateWithGrads {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}
```

From this flexible trait we get:
1. Single & batched inputs (just have multiple impls!)
2. Update with gradients
3. Multiple inputs/outputs (multi-headed modules, or rnns)
4. Behavior different when tape is present or not (**not** the .train()/.eval() behavior present in other libraries!).

### Tuples represent feedforward (a.k.a sequential) modules

Since we can implement traits for tuples, which is *not possible in other languages* AFAIK, they provide a very nice frontend
for sequentially executing modules.

```rust
// no idea why you would do this, but you could!
let model: (ReLU, Sigmoid, Tanh) = Default::default();
```

```rust
let model: (Linear<10, 5>, Tanh) = Default::default();
```

How implementing Module for a 2-tuple looks:
```rust
impl<Input, A, B> Module<Input> for (A, B)
where
    Input: Tensor,
    A: Module<Input>,        // A is a module that takes Input
    B: Module<A::Output>,    // B is a module that takes A's Output
{
    type Output = B::Output; // the output of this is B's Output
    fn forward(&self, x: Input) -> Self::Output {
        let x = self.0.forward(x);
        let x = self.1.forward(x);
        x
    }
}
```

We've implemented Module for Tuples up to 6 elements, but *you can arbitrarily nest them*!

### No Rc/RefCells used - Gradient tape is not reference counted!

Other implementations may store a reference to the gradient tape directly on tensors, which requires mutating tensors or using Rc/Refcells all other the place.

We've figured out an elegant way to avoid this, reducing references and dynamic reference count checks to 0!

Since all operations result in exactly 1 child, we can always move the gradient tape to the child of the last operation. Additionally, no model parameters (all tensors) will ever own the gradient tape because they will never be the result of any operation. This means we know exactly which tensor owns the gradient tape, and the tensors that have it will always be intermediate results that don't need to be maintained across gradient computation.

*All of this together gives users unprecedented control/precision over what tensors are recorded on the gradient tape!*

One advanced use case requires that tensors be re-used multiple times in a computation graph.
This can be handled by duplicating the tensor, and manually moving the gradient tape around.
See [examples/multi_head.rs](examples/multi_head.rs) for an example.

### Type checked backward

tl;dr: If you forget to include a call to `trace()` or `traced()`, the program won't compile!

```diff
-let pred = module.forward(x);
+let pred = module.forward(x.traced());
let loss = (&y - pred).square().mean();
let gradients = loss.backward();
```

Since we know exactly what tensors own the gradient tape, we can require the tensor passed into `.backward()` to own the gradient tape!
And further, we can require it be moved into `.backward()`, so it can destruct the tape and construct the gradients!

__All of this can be checked at compile time 🎉__

```rust
pub fn backward<T: Tensor<Tape = OwnedTape>>(t: T) -> Gradients {
    let (t, tape): (T::NoTape, OwnedTape) = t.split_tape();
    tape.0.backward(&t)
}
```

### Recursive trait definitions for CPU Device

Our [src/devices](src/devices/) backend for computing operations on the CPU
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

Encourage you to check out all the code in [src/devices](src/devices/)!

### 📄 Validated against pytorch

All functions & operations are tested against behavior shown by similar code in pytorch.

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
