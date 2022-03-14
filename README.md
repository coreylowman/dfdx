# STAG: Strongly Typed Auto Grad

Reverse Mode Auto Differentiation[1] in Rust.

```rust
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Uniform;
use stag::prelude::*;

type MLP = (
    (Linear<10, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 2>, Tanh),
);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let module: MLP = Default::default();
    let x: Tensor2D<64, 10> = Tensor2D::randn(&mut rng);
    let y = module.forward(&x);
}
```

[1] https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation


## Features

1. 👌 Simple Neural Networks API, completely type checked at compile time. See [examples/regression.rs](examples/regression.rs)
2. 📈 Easy to use Optimizer API
3. 💡 Tensor sizes & operations type checked at compile time
4. ✔ Unsafe free rust code

## Fun/notable implementation details

### Module

The Module trait is simple and powerful!

```rust
pub trait Module<I>: Default + HasGradients {
    type Output;
    fn forward(&self, input: &I) -> Self::Output;
}
```

The generic type variable for the input allows a single struct to implement module for multiple inputs. *This is how batching is implented!*

### Feedforward (a.k.a Sequential) modules & Tuples

The associated Output type makes combining multiple modules to be executed sequentially very easy:

```rust
impl<Input, A, B> Module<Input> for (A, B)
where
    Input: Tensor,
    A: Module<Input>,
    B: Module<A::Output>,
{
    type Output = B::Output;
    fn forward(&self, x: &Input) -> Self::Output {
        let x = self.0.forward(x);
        self.1.forward(&x)
    }
}
```

What's also nice is that we can use tuple's as the container instead of introducing a whole new struct. *This is not possible in other languages!*

### Gradient tape is not reference counted!

Since all operations result in exactly 1 child, we can always move the gradient tape to the child of the last operation. This means we know exactly which tensor owns the gradient tape!
