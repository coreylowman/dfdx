use crate::{
    shapes::*,
    tensor::*,
    tensor_ops::{Device, TryAdd},
};

/// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function). `1 / (1 + exp(-t))`.
///
/// The derivative is `sigmoid(t) * (1.0 - sigmoid(t))`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let r = t.sigmoid();
/// ```
pub fn sigmoid<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T> {
    t.sigmoid()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    /// See [sigmoid]
    pub fn sigmoid(self) -> Self {
        self.try_sigmoid().unwrap()
    }
    /// See [sigmoid]
    pub fn try_sigmoid(self) -> Result<Self, D::Err> {
        self.try_negate()?.try_exp()?.try_add(E::ONE)?.try_recip()
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_sigmoid() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().sigmoid();
        assert_close(
            &r.array(),
            &[0.11920292, 0.26894143, 0.5, 0.7310586, 0.880797],
        );
        let g = r.mean().backward();
        assert_close(
            &g.get(&x).array(),
            &[0.020998716, 0.039322387, 0.05, 0.039322387, 0.020998726],
        );
    }
}
