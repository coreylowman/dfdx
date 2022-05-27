//! Standard loss functions such as mse, mae, cross entropy, and more.

use crate::prelude::*;

/// Mean Squared Error. This is the same as doing `(pred - &targ).square().mean()`
pub fn mse_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    mean(square(sub(targ, pred)))
}

/// Root Mean square error. This is the same as doing `(pred - &targ).square().mean().sqrt()`
pub fn rmse_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    sqrt(mse_loss(pred, targ))
}

/// Mean absolute error. This is the same as doing `(pred - &targ).abs().mean()`
pub fn mae_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    mean(abs(sub(targ, pred)))
}

/// Cross entropy loss. This will call `log_softmax(logits)`, so make sure logits is **not** the
/// output from [softmax()] or [log_softmax()] already.
///
/// This computes: `-(logits.log_softmax() * targ).sum(-1).mean()`
///
/// Arguments:
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called in this function
/// - `targ`: Target containing probability vectors **NOT** class indices.
///
/// Example Usage:
/// ```rust
/// # use dfdx::prelude::*;
/// let x = Tensor1D::new([-1.0, -0.5]);
/// let targ = Tensor1D::new([0.5, 0.5]);
/// let loss = cross_entropy_with_logits_loss(x.traced(), &targ);
/// ```
pub fn cross_entropy_with_logits_loss<T: Tensor<Dtype = f32>>(
    logits: T,
    targ: &T::NoTape,
) -> Tensor0D<T::Tape> {
    -mean(sum_last_dim(mul(targ, log_softmax(logits))))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let x = Tensor1D::new([0.87248087, -0.24252531, -1.00609493, 1.15508401, 1.55450475]);
        let y = Tensor1D::new([
            -0.90954804,
            -1.01931846,
            -0.39221755,
            2.25248861,
            1.30355537,
        ]);
        let loss = mse_loss(x.trace(), &y);
        assert_eq!(loss.data(), &1.0846305);
        let g = loss.backward();
        assert_eq!(
            g.ref_gradient(&x),
            &[0.71281159, 0.31071725, -0.24555098, -0.43896183, 0.10037976]
        );
    }

    #[test]
    fn test_mae() {
        let x = Tensor1D::new([0.87248087, -0.24252531, -1.00609493, 1.15508401, 1.55450475]);
        let y = Tensor1D::new([
            -0.90954804,
            -1.01931846,
            -0.39221755,
            2.25248861,
            1.30355537,
        ]);
        let loss = mae_loss(x.trace(), &y);
        assert_eq!(loss.data(), &0.90421069);
        let g = loss.backward();
        assert_eq!(g.ref_gradient(&x), &[0.2, 0.2, -0.2, -0.2, 0.2]);
    }

    #[test]
    fn test_soft_cross_entropy() {
        let x = Tensor1D::new([-0.57227212, 0.84696430, 1.20634139, -1.09643006, 1.19451940]);
        let y = Tensor1D::new([0.10473672, 0.24449949, 0.32667059, 0.22253996, 0.10155323]);
        let loss = cross_entropy_with_logits_loss(x.trace(), &y);
        assert_eq!(loss.data(), &1.8713832);
        let g = loss.backward();
        assert_eq!(
            g.ref_gradient(&x),
            &[
                -0.047592897,
                -0.008269042,
                0.0117146075,
                -0.18870775,
                0.23285514
            ]
        );
    }

    #[test]
    fn test_crossentropy() {
        let x = Tensor1D::new([0.87248087, -0.24252531, -1.00609493, 1.15508401, 1.55450475]);
        let losses = [1.56552291, 2.68052912, 3.4440987, 1.2829198, 0.883499];
        for i in 0..5 {
            let mut targ = [0.0; 5];
            targ[i] = 1.0;
            let y = Tensor1D::new(targ);
            let loss = cross_entropy_with_logits_loss(x.trace(), &y);
            assert_eq!(*loss.data(), losses[i]);
        }
    }
}
