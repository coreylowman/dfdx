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
    let x = Tensor2D::<64, 10>::randn(&mut rng);
    let y = Tensor2D::<64, 2>::randn(&mut rng);

    // initialize our MLP with uniform random weights
    let mut module: MLP = Default::default();
    module.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    // trace x through the module
    x.with_grad();
    let pred = module.forward(&x);
    let loss = sub(&pred, &y).square().mean();

    // compute gradients
    let mut sgd = Sgd { lr: 1e-2 };
    let gradients = sgd.compute_gradients(&loss);

    // update the MLP
    module.update_with_gradients(&gradients);
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

### Gradient tape is not reference counted!

Since all operations result in exactly 1 child, we can always move the gradient tape to the child of the last operation. This means we know exactly which tensor owns the gradient tape!
