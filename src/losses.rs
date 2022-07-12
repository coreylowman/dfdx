//! Standard loss functions such as [mse_loss()], [cross_entropy_with_logits_loss()], and more.

use crate::prelude::*;

/// [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error).
/// This computes `(&targ - pred).square().mean()`.
///
/// See [mean()], [square()], and [sub()].
pub fn mse_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    mean(square(sub(pred, targ)))
}

/// [Root Mean square error](https://en.wikipedia.org/wiki/Root-mean-square_deviation).
/// This computes `(&targ - pred).square().mean().sqrt()`
///
/// See [mse_loss()] and [sqrt()]
pub fn rmse_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    sqrt(mse_loss(pred, targ))
}

/// [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error).
/// This computes `(&targ - pred).abs().mean()`
///
/// See [mean()], [abs()], and [sub()]
pub fn mae_loss<T: Tensor<Dtype = f32>>(pred: T, targ: &T::NoTape) -> Tensor0D<T::Tape> {
    mean(abs(sub(pred, targ)))
}

/// [Cross entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression).
/// This computes: `-(logits.log_softmax() * target_probs).sum(-1).mean()`
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
    -mean(sum_last_dim(mul(log_softmax(logits), target_probs)))
}

/// [KL Divergence loss](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
/// This computes `(target_probs * (target_probs.log() - logits.log_softmax())).sum(-1).mean()`
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
    -mean(sum_last_dim(mul(
        sub(log_softmax(logits), &ln(target_probs.duplicate())),
        target_probs,
    )))
}

/// [Binary Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression) With Logits in numerically stable way.
///
/// Computes `(1 - target_probs) * logits + log(1 + exp(-logits))`.
///
/// Inputs:
/// - `logits` - unnormalized inputs. **NOT** output of sigmoid
/// - `target_probs` - target values between 0 and 1.
///
/// Implementation Details:
///
/// The numerical stable version involves subtracting the maximum value
/// of logits in a way that doesn't change the output (i.e. adding zero):
/// 1. Start with `log(1 + exp(-logits))`
/// 2. Add zero (variable Q can be anything) `Q - Q + log(1 + exp(-logits))`
/// 3. log(exp(Q)) == Q: `Q - log(exp(Q)) + log(1 + exp(-logits))`
/// 4. log(x) - log(y) = log(x / y): `Q + log((1 + exp(-logits)) / exp(Q))`
/// 5. 1 / exp(Q) = exp(-Q): `Q + log(exp(-Q) + exp(-logits) / exp(Q))`
/// 6. exp(A) / exp(B) = exp(A-B): `Q + log(exp(-Q) + exp(-logits - Q))`
/// 7. Now set `Q = max(logits)`!
pub fn binary_cross_entropy_with_logits_loss<T: Tensor<Dtype = f32>>(
    logits: T,
    target_probs: &T::NoTape,
) -> Tensor0D<T::Tape> {
    let (logits, tape) = logits.split_tape();

    // max_value = (-logits).clamp(min=0)
    let max_value = clamp(negate(logits.duplicate()), 0.0, f32::INFINITY);

    // a = exp(-(max_value + logits))
    let a = exp(negate(add(logits.duplicate().put_tape(tape), &max_value)));

    // b = ln(exp(-max_value) + a)
    let b = ln(add(a, &exp(negate(max_value.duplicate()))));

    // c = max_value + b
    let c = add(b, &max_value);
    let (c, tape) = c.split_tape();

    // d = (1 - T)
    let d = negate(sub_scalar(target_probs.duplicate(), 1.0));

    // e = logits * d
    let e = mul(logits.put_tape(tape), &d);

    mean(add(e, &c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let x = Tensor1D::new([0.87248087, -0.24252531, -1.0060949, 1.155084, 1.5545048]);
        let y = Tensor1D::new([-0.90954804, -1.0193185, -0.39221755, 2.2524886, 1.3035554]);
        let loss = mse_loss(x.trace(), &y);
        assert_eq!(loss.data(), &1.0846305);
        let g = loss.backward();
        assert_eq!(
            g.ref_gradient(&x),
            &[0.7128116, 0.31071725, -0.24555098, -0.43896183, 0.10037976]
        );
    }

    #[test]
    fn test_mae() {
        let x = Tensor1D::new([0.87248087, -0.24252531, -1.0060949, 1.155084, 1.5545048]);
        let y = Tensor1D::new([-0.90954804, -1.0193186, -0.39221755, 2.2524886, 1.3035554]);
        let loss = mae_loss(x.trace(), &y);
        assert_eq!(loss.data(), &0.9042107);
        let g = loss.backward();
        assert_eq!(g.ref_gradient(&x), &[0.2, 0.2, -0.2, -0.2, 0.2]);
    }

    #[test]
    fn test_soft_cross_entropy() {
        let x = Tensor1D::new([-0.5722721, 0.8469643, 1.2063414, -1.0964301, 1.1945194]);
        let y = Tensor1D::new([0.10473672, 0.24449949, 0.3266706, 0.22253996, 0.10155323]);
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
        let x = Tensor1D::new([0.87248087, -0.24252531, -1.0060949, 1.155084, 1.5545048]);
        let losses = [1.5655229, 2.680529, 3.4440987, 1.2829198, 0.883499];
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

    #[test]
    fn test_bce() {
        let p = Tensor2D::new([[100.0; 3], [-100.0; 3], [-1.0, 0.0, 1.0]]);
        let t = Tensor2D::new([[0.0, 0.5, 1.0]; 3]);

        let loss = binary_cross_entropy_with_logits_loss(p.trace(), &t);
        assert_eq!(loss.data(), &33.479965);

        let gradients = loss.backward();

        assert_eq!(
            gradients.ref_gradient(&p),
            &[
                [0.11111111, 0.055555556, -4e-45],
                [0.0, -0.055555556, -0.11111111],
                [0.029882379, 0.0, -0.02988238]
            ]
        );
    }
}
