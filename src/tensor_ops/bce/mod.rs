mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::{try_binary_op, BinaryKernel};
use crate::{gradients::*, shapes::*, tensor::Tensor};

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
    LTape: Tape<D> + Merge<RTape>,
    RTape: Tape<D>,
{
    logits.bce_with_logits(probs)
}

impl<S: Shape, E: Dtype, D: BinaryKernel<BCEKernelOp, E>, LTape: Tape<D>> Tensor<S, E, D, LTape> {
    /// See [bce_with_logits]
    pub fn bce_with_logits<RTape: Tape<D>>(self, prob: Tensor<S, E, D, RTape>) -> Self
    where
        LTape: Merge<RTape>,
    {
        self.try_bce_with_logits(prob).unwrap()
    }
    /// See [bce_with_logits]
    pub fn try_bce_with_logits<RTape>(self, prob: Tensor<S, E, D, RTape>) -> Result<Self, D::Err>
    where
        RTape: Tape<D>,
        LTape: Merge<RTape>,
    {
        try_binary_op(BCEKernelOp, self, prob)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tensor::*,
        tests::{assert_close, TestDevice},
    };

    #[test]
    fn test_bce() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([
            [-0.84240317, 0.63094819, 1.04164326],
            [1.32522500, 0.58402753, 1.91676331],
        ]);
        let b = dev.tensor([
            [0.52022195, 0.57880402, 0.17535722],
            [0.75429636, 0.66566986, 0.61827511],
        ]);
        let r = a.trace().bce_with_logits(b);
        assert_close(
            &r.array(),
            &[
                [0.79638255, 0.69238377, 1.161215],
                [0.56127203, 0.63843495, 0.8688978],
            ],
        );
    }
}
