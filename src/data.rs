use crate::prelude::*;

/// One hot encodes an array of class labels into a [Tensor2D] of probability
/// vectors. This can be used in tandem with [cross_entropy_with_logits_loss()].
///
/// Const Generic Arguments:
/// - `B` - the batch size
/// - `N` - the number of classes
///
/// Arguments:
/// - `class_labels` - an array of size `B` where each element is the class label
///
/// Outputs: [Tensor2D] with shape (B, N)
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
///
/// let class_labels = [0, 1, 2, 1, 1];
/// // NOTE: 5 is the batch size, 3 is the number of classes
/// let probs = one_hot_encode::<5, 3>(&class_labels);
/// assert_eq!(probs.data(), &[
///     [1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 1.0, 0.0],
/// ]);
/// ```
pub fn one_hot_encode<const B: usize, const N: usize>(class_labels: &[usize; B]) -> Tensor2D<B, N> {
    let mut result = Tensor2D::zeros();
    for i in 0..B {
        result.mut_data()[i][class_labels[i]] = 1.0;
    }
    result
}
