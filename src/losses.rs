//! Standard loss functions such as mse, mae, cross entropy, and more.

use crate::prelude::*;

/// Mean Squared Error. This computes `(&targ - pred).square().mean()`.
///
/// See [mean()], [square()], and [sub()].
pub fn mse_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    mean(square(sub(targ, pred)))
}

/// Root Mean square error. This computes `(&targ - pred).square().mean().sqrt()`
///
/// See [mse_loss()] and [sqrt()]
pub fn rmse_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    sqrt(mse_loss(pred, targ))
}

/// Mean absolute error. This computes `(&targ - pred).abs().mean()`
///
/// See [mean()], [abs()], and [sub()]
pub fn mae_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    mean(abs(sub(targ, pred)))
}

/// Cross entropy loss. This computes: `-(logits.log_softmax() * target_probs).sum(-1).mean()`
///
/// This will call `log_softmax(logits)`, so make sure logits is **not** the
/// output from [softmax()] or [log_softmax()] already.
///
/// Arguments:
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called **in** this function
/// - `target_probs`: Target containing probability vectors **NOT** class indices.
///
/// Example Usage:
/// ```rust
/// # use dfdx::prelude::*;
/// let x = Tensor1D::new([-1.0, -0.5]);
/// let target_probs = Tensor1D::new([0.5, 0.5]);
/// let loss = cross_entropy_with_logits_loss(x.traced(), &target_probs);
/// ```
pub fn cross_entropy_with_logits_loss<T: Tensor<Dtype = f32>>(
    logits: T,
    target_probs: &T::NoTape,
) -> Tensor0D<T::Tape> {
    -mean(sum_last_dim(mul(target_probs, log_softmax(logits))))
}

/// KL Divergence loss. This computes `(target_probs * (target_probs.log() - logits.log_softmax())).sum(-1).mean()`
///
/// This will call `log_softmax(logits)`, so make sure logits is **not** the
/// output from [softmax()] or [log_softmax()] already.
///
/// Arguments:
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called **in** this function
/// - `target_probs`: Target containing probability vectors **NOT** class indices.
///
/// Example Usage:
/// ```rust
/// # use dfdx::prelude::*;
/// let x = Tensor1D::new([-1.0, -0.5]);
/// let target_probs = Tensor1D::new([0.5, 0.5]);
/// let loss = kl_div_with_logits_loss(x.traced(), &target_probs);
/// ```
pub fn kl_div_with_logits_loss<T: Tensor<Dtype = f32>>(
    logits: T,
    target_probs: &T::NoTape,
) -> Tensor0D<T::Tape> {
    mean(sum_last_dim(mul(
        target_probs,
        sub(&ln(target_probs.clone()), log_softmax(logits)),
    )))
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

    #[test]
    fn test_kl_div() {
        let logits = Tensor2D::new([
            [-0.2354, 0.4408, 0.9688],
            [-0.2187, -0.3451, -1.5473],
            [0.7420, 0.7186, 1.0785],
            [-1.2231, 0.2536, 0.3489],
            [-0.9163, -0.2289, 0.2576],
        ]);
        let targ = Tensor2D::new([
            [0.3178, 0.5344, 0.1479],
            [0.1915, 0.6178, 0.1907],
            [0.4834, 0.1789, 0.3377],
            [0.5809, 0.3623, 0.0568],
            [0.0166, 0.8512, 0.1322],
        ]);
        let loss = kl_div_with_logits_loss(logits.trace(), &targ);
        assert_eq!(loss.data(), &0.40656146);
        let gradients = loss.backward();
        assert_eq!(
            gradients.ref_gradient(&logits),
            &[
                [-0.031813223, -0.044453412, 0.07626665],
                [0.05489187, -0.04143352, -0.013458336],
                [-0.037454266, 0.02207594, 0.015378334],
                [-0.09656205, 0.013436668, 0.083125375],
                [0.02881821, -0.10633193, 0.0775137]
            ]
        );
    }
}
