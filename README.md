# dfdx: shape checked deep learning in rust

[![CUDA](https://badgen.net/badge/CUDA/passing/green)](#)
[![crates.io](https://img.shields.io/crates/v/dfdx.svg)](https://crates.io/crates/dfdx)
[![docs.rs](https://img.shields.io/docsrs/dfdx)](https://docs.rs/dfdx)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/AtUhGqBDP5)

Ergonomics & safety focused deep learning in Rust.

**Still in pre-alpha state. The next few releases are planned to be breaking releases.**

Features at a glance:
1. :fire: GPU accelerated tensor library with shapes up to 6d!
2. Shapes with both compile and runtime sized dimensions. (e.g. `Tensor<(usize, Const<10>)>` and `Tensor<Rank2<5, 10>>`)
3. A large library of tensor operations (including `matmul`, `conv2d`, and much more).
    1. All tensor operations shape and type checked at compile time!!
4. Ergonomic neural network building blocks (like `Linear`, `Conv2D`, and `Transformer`).
5. Standard deep learning optimizers such as `Sgd`, `Adam`, `AdamW`, `RMSprop`, and more.

`dfdx` is on [crates.io](https://crates.io/crates/dfdx)! Use by adding this to your `Cargo.toml`:

```toml
dfdx = "0.10.0"
```

See the documentation at [docs.rs/dfdx](https://docs.rs/dfdx).

[1] https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation

## Design Goals

1. Ergonomics the whole way down (both frontend interface & internals).
2. Check as much at compile time as possible (i.e. don't compile if something is not correct).
3. Maximize performance.
4. Minimize unsafe code[1]
5. Minimize Rc<RefCell<T>> used in internal code[2]

[1] Currently the only unsafe calls are for matrix multiplication.

[2] The only things that use `Arc` are tensors to store their data. `Arc` is used instead of `Box` to reduce
allocations when tensors are cloned.

## GPU acceleration with CUDA

Enable the `cuda` feature to start using the `Cuda` device! Requires the installation of nvidia's cuda toolkit. See [feature flags docs](https://docs.rs/dfdx/latest/dfdx/feature_flags/index.html) for more info.

## BLAS libraries

The [matrixmultiply crate](https://crates.io/crates/matrixmultiply) is the default BLAS library. **You don't need
to do download/install anything for this to work!**

To link to the `Intel MKL` libraries (assuming you installed it already) use the `cpu-mkl-matmul` feature. See [feature flags docs](https://docs.rs/dfdx/latest/dfdx/feature_flags/index.html) for more info.

## API Preview

Check [examples/](examples/) for more details.

1. 👌 Simple Neural Networks API, completely shape checked at compile time.

```rust
type Mlp = (
    (Linear<10, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() {
    let dev: Cuda = Default::default(); // or `Cpu`
    let mlp = dev.build_module::<Mlp, f32>();
    let x: Tensor<Rank1<10>, f32, Cpu> = dev.zeros();
    let y: Tensor<Rank1<2>, f32, Cpu> = mlp.forward(x);
    mlp.save("checkpoint.npz")?;
}
```

2. 📈 Ergonomic Optimizer API

```rust
type Model = ...
let mut model = dev.build_module::<Model, f32>();
let mut grads = model.alloc_grads();
let mut sgd = Sgd::new(&model, SgdConfig {
    lr: 1e-2,
    momentum: Some(Momentum::Nesterov(0.9))
});

let loss = ...
grads = loss.backward();

sgd.update(&mut model, &grads);
```

3. 💡 Const tensors can be converted to and from normal rust arrays
```rust
let t0: Tensor<Rank0, f32, _> = dev.tensor(0.0);
assert_eq!(t0.array(), &0.0);

let t1 /*: Tensor<Rank1<3>, f32, _>*/ = dev.tensor([1.0, 2.0, 3.0]);
assert_eq!(t1.array(), [1.0, 2.0, 3.0]);

let t2: Tensor<Rank2<2, 3>, f32, _> = dev.sample_normal();
assert_ne!(t2.array(), [[0.0; 3]; 2]);
```

## Fun/notable implementation details

### Module

```rust
pub trait Module<Input> {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}
```

From this flexible trait we get:
1. Single & batched inputs (just have multiple impls!)
2. Multiple inputs/outputs (multi-headed modules, or rnns)
3. Behavior different when tape is present or not (**not** the .train()/.eval() behavior present in other libraries!).

### Tuples represent feedforward (a.k.a sequential) modules

Since we can implement traits for tuples, which is *not possible in other languages* AFAIK, they provide a very nice frontend
for sequentially executing modules.

```rust
// no idea why you would do this, but you could!
type Model = (ReLU, Sigmoid, Tanh);
let model = dev.build_module::<Model, f32>();
```

```rust
type Model = (Linear<10, 5>, Tanh)
let model = dev.build_module::<Model, f32>();
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

Modules implemented for Tuples up to 6 elements, but *you can arbitrarily nest them*!

### No `Rc<RefCells<T>>` used - Gradient tape is not kept behind a cell!

Other implementations may store a reference to the gradient tape directly on tensors, which requires mutating tensors or using Rc/Refcells all over the place.

We've figured out an elegant way to avoid this, reducing references and dynamic borrow checks to 0!

Since all operations result in exactly 1 child, we can always move the gradient tape to the child of the last operation. Additionally, no model parameters (all tensors) will ever own the gradient tape because they will never be the result of any operation. This means we know exactly which tensor owns the gradient tape, and the tensors that have it will always be intermediate results that don't need to be maintained across gradient computation.

*All of this together gives users unprecedented control/precision over what tensors are recorded on the gradient tape!*

One advanced use case requires that tensors be re-used multiple times in a computation graph.
This can be handled by cloning the tensor, and manually moving the gradient tape around.

### Type checked backward

tl;dr: If you forget to include a call to `trace()` or `traced()`, the program won't compile!

```diff
-let pred = module.forward(x);
+let pred = module.forward(x.traced(grads));
let loss = (y - pred).square().mean();
let gradients = loss.backward();
```

Since we know exactly what tensors own the gradient tape, we can require the tensor passed into `.backward()` to own the gradient tape!
And further, we can require it be moved into `.backward()`, so it can destruct the tape and construct the gradients!

__All of this can be checked at compile time 🎉__

### 📄 Validated against pytorch

All functions & operations are tested against behavior shown by similar code in pytorch.

# License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
