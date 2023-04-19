mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_binary_op, BinaryKernel};
use crate::{shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BCEKernelOp;

/// [Binary Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression) With Logits in numerically stable way.
///
/// Computes `target_probs * log(sigmoid(logits)) + (1 - target_probs) * log(1 - sigmoid(logits))`
/// as `(1 - target_probs) * logits + log(1 + exp(-logits))`.
///
/// # Inputs
/// - `logits` - unnormalized inputs. **NOT** output of sigmoid
/// - `target_probs` - target values between 0 and 1.
///
/// # Numerically Stable Derivation
///
/// See <https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits>
/// for more information on this.
pub fn bce_with_logits<S: Shape, E: Dtype, D: BinaryKernel<BCEKernelOp, E>, LTape, RTape>(
    logits: Tensor<S, E, D, LTape>,
    probs: Tensor<S, E, D, RTape>,
) -> Tensor<S, E, D, LTape>
where
    LTape: Tape<E, D> + Merge<RTape>,
    RTape: Tape<E, D>,
{
    logits.bce_with_logits(probs)
}

impl<S: Shape, E: Dtype, D: BinaryKernel<BCEKernelOp, E>, LTape: Tape<E, D>>
    Tensor<S, E, D, LTape>
{
    /// See [bce_with_logits]
    pub fn bce_with_logits<RTape: Tape<E, D>>(self, prob: Tensor<S, E, D, RTape>) -> Self
    where
        LTape: Merge<RTape>,
    {
        self.try_bce_with_logits(prob).unwrap()
    }
    /// See [bce_with_logits]
    pub fn try_bce_with_logits<RTape>(self, prob: Tensor<S, E, D, RTape>) -> Result<Self, D::Err>
    where
        RTape: Tape<E, D>,
        LTape: Merge<RTape>,
    {
        try_binary_op(BCEKernelOp, self, prob)
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::*};

    #[test]
    fn test_bce() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, TestDtype, _> = dev.tensor([
            [-0.8424031, 0.6309481, 1.0416432],
            [1.325225, 0.5840275, 1.9167633],
        ]);
        let b: Tensor<_, TestDtype, _> = dev.tensor([
            [0.52022195, 0.578804, 0.17535722],
            [0.75429636, 0.66566986, 0.6182751],
        ]);
        let r = a.leaky_trace().bce_with_logits(b);
        assert_close_to_literal!(
            r,
            [
                [0.79638255, 0.69238377, 1.161215],
                [0.561272, 0.63843495, 0.8688978],
            ]
        );
    }
}
