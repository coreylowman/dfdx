//! Standard loss functions such as mse, mae, cross entropy, and more.

use crate::prelude::*;

/// Mean Squared Error. This is the same as doing `(pred - &targ).abs().mean()`
pub fn mse_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::TapeHolder> {
    sub(targ, pred).square().mean()
}

/// Mean absolute error. This is the same as doing `(pred - &targ).abs().mean()`
pub fn mae_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::TapeHolder> {
    sub(targ, pred).abs().mean()
}

/// Cross entropy loss. This will call `logits.log_softmax()`, so make sure logits is not the
/// output from softmax() or log_softmax() already.
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x = Tensor1D::new([-1.0, -0.5]);
/// let targ = Tensor1D::new([0.5, 0.5]);
/// let loss = cross_entropy_with_logits_loss(x.traced(), &targ);
/// ```
///
/// `targ` is a soft target (i.e. each of the sub tensors is a probability vector).
///
/// If you're looking to use a hard target (i.e. an index),
/// use this implementaiton with .sum() instead of .mean() at the end.
pub fn cross_entropy_with_logits_loss<T: Tensor<Dtype = f32> + HasSoftmaxMethod>(
    logits: T,
    targ: &T::NoTape,
) -> Tensor0D<T::TapeHolder> {
    -mul(targ, logits.log_softmax()).mean()
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
    fn test_crossentropy() {
        let x = Tensor1D::new([0.87248087, -0.24252531, -1.00609493, 1.15508401, 1.55450475]);
        let losses = [1.56552291, 2.68052912, 3.444099, 1.2829198, 0.883499];
        for i in 0..5 {
            let mut targ = [0.0; 5];
            targ[i] = 1.0;
            let y = Tensor1D::new(targ);
            let loss = cross_entropy_with_logits_loss(x.trace(), &y);
            assert_eq!(*loss.data() * 5.0, losses[i]);
        }
    }
}
