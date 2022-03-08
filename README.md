# STAG: Strongly Typed Auto Grad

Reverse Mode Auto Differentiation[1] in Rust.

[1] https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation


## 👌 Simple Neural Networks API, completely type checked at compile time.

See [examples/linear.rs](examples/linear.rs) and [examples/regression.rs](examples/regression.rs) for more examples.

```rust
TODO
```

## 💡 Tensor sizes & operations type checked at compile time

See [examples/tensor.rs](examples/tensor.rs) for more tensor operation examples.

```rust
TODO
```

## 📄 Batching completely supported by type system

Since the `Module` trait is generic, we can implement module's for multiple inputs/outputs.
To support batching all we have to do is impl Module a second time with a batch dimension
added to inputs/outputs!

See [src/nn/linear.rs](src/nn/linear.rs) for an example implementation.

TODO is this still accurate?
NOTE: Unfortunately because of the ModuleChain currently works, a model constructed
using ModuleChain can't call forward with two different data types.

```rust
TODO
```

## 📈 Easy to use Optimizer API

See [examples/regression.rs](examples/regression.rs) for more examples.

```rust
TODO
```

## Interesting implementation details

### No Arc/RefCell (& why tensors require mutability)

Since all operations in a computation graph have exactly 1 child, we can always move the gradient tape to the child of the last operation. This means we know exactly which tensor owns the gradient tape!

This is also why we have to mark all the tensors as mut and pass them around with &mut to the operations. Every operation could potentially pull the gradient tape out of the tensor!.

### Module & ModuleChain

I'm partial to the Module trait:

```rust
TODO
```
This is nice because we can impl Module for different inputs for the same struct, which is how batching is implemented!

This also enables an easy sequential/chaining functionality with tuples, by implementing Module for a tuple of modules:

```rust
TODO
```

### Optimizer has all methods of underlying module for free!

Yay [DerefMut](https://doc.rust-lang.org/std/ops/trait.DerefMut.html)!

