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

/// [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < delta`: `0.5 * (x - y)^2`
/// 2. otherwise: `delta * (|x - y| - 0.5 * delta)`
///
/// # Example
/// ```rust
/// # use dfdx::prelude::*;
/// let x = Tensor1D::new([-1.0, -0.5]);
/// let y = Tensor1D::new([0.5, 0.5]);
/// let loss = huber_loss(x.traced(), &y, 1.0);
/// ```
pub fn huber_loss<T: Tensor<Dtype = f32>>(
    pred: T,
    targ: &T::NoTape,
    delta: T::Dtype,
) -> Tensor0D<T::Tape> {
    let f = move |x: &f32, y: &f32| {
        if (x - y).abs() < delta {
            (x - y).powi(2) * 0.5
        } else {
            (x - y).abs() * delta - 0.5 * delta * delta
        }
    };
    let dfdx = move |x: &f32, y: &f32| {
        if (x - y) == 0.0 {
            0.0
        } else if (x - y).abs() < delta {
            x - y
        } else {
            (x - y).signum() * delta
        }
    };
    let dfdy = move |x: &f32, y: &f32| {
        if (x - y) == 0.0 {
            0.0
        } else if (x - y).abs() < delta {
            y - x
        } else {
            (y - x).signum() * delta
        }
    };
    mean(crate::tensor_ops::utils::binary_map(
        pred, targ, f, dfdx, dfdy,
    ))
}

/// Smooth l1 loss (closely related to [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss))
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < beta`: `0.5 * (x - y)^2 / beta`
/// 2. otherwise: `|x - y| - 0.5 * beta`
///
/// # Example
/// ```rust
/// # use dfdx::prelude::*;
/// let x = Tensor1D::new([-1.0, -0.5]);
/// let y = Tensor1D::new([0.5, 0.5]);
/// let loss = smooth_l1_loss(x.traced(), &y, 1.0);
/// ```
pub fn smooth_l1_loss<T: Tensor<Dtype = f32>>(
    pred: T,
    targ: &T::NoTape,
    beta: T::Dtype,
) -> Tensor0D<T::Tape> {
    huber_loss(pred, targ, beta) / beta
}

/// [Cross entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression).
/// This computes: `-(logits.log_softmax() * target_probs).sum(-1).mean()`
///
/// This will call `log_softmax(logits)`, so make sure logits is **not** the
/// output from [softmax()] or [log_softmax()] already.
///
/// # Arguments
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called **in** this function
/// - `target_probs`: Target containing probability vectors **NOT** class indices.
///
/// # Example
/// ```rust
/// # use dfdx::prelude::*;
/// let logits = Tensor1D::new([-1.0, -0.5]);
/// let target_probs = Tensor1D::new([0.5, 0.5]);
/// let loss = cross_entropy_with_logits_loss(logits.traced(), &target_probs);
/// ```
pub fn cross_entropy_with_logits_loss<T: Reduce<Axis<-1>>>(
    logits: T,
    target_probs: &T::NoTape,
) -> Tensor0D<T::Tape> {
    -mean(sum_axis(mul(log_softmax(logits), target_probs)))
}

/// [KL Divergence loss](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
/// This computes `(target_probs * (target_probs.log() - logits.log_softmax())).sum(-1).mean()`
///
/// This will call `log_softmax(logits)`, so make sure logits is **not** the
/// output from [softmax()] or [log_softmax()] already.
///
/// # Arguments
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called **in** this function
/// - `target_probs`: Target containing probability vectors **NOT** class indices.
///
/// # Example
/// ```rust
/// # use dfdx::prelude::*;
/// let logits = Tensor1D::new([-1.0, -0.5]);
/// let target_probs = Tensor1D::new([0.5, 0.5]);
/// let loss = kl_div_with_logits_loss(logits.traced(), &target_probs);
/// ```
pub fn kl_div_with_logits_loss<T: Reduce<Axis<-1>>>(
    logits: T,
    target_probs: &T::NoTape,
) -> Tensor0D<T::Tape> {
    -mean(sum_axis(mul(
        sub(log_softmax(logits), &ln(target_probs.duplicate())),
        target_probs,
    )))
}

/// [Binary Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression) With Logits in numerically stable way.
///
/// Computes `target_probs * log(sigmoid(logits)) + (1 - target_probs) * log(1 - sigmoid(logits))`
/// as `(1 - target_probs) * logits + log(1 + exp(-logits))`.
///
/// # Inputs
/// - `logits` - unnormalized inputs. **NOT** output of sigmoid
/// - `target_probs` - target values between 0 and 1.
///
/// # Example
/// ```rust
/// # use dfdx::prelude::*;
/// let logits = Tensor1D::new([-1.0, -0.5]);
/// let target_probs = Tensor1D::new([1.0, 0.25]);
/// let loss = binary_cross_entropy_with_logits_loss(logits.traced(), &target_probs);
/// ```
///
/// # Numerically Stable Derivation
///
/// See <https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits>
/// for more information on this.
pub fn binary_cross_entropy_with_logits_loss<T: Tensor<Dtype = f32>>(
    logits: T,
    target_probs: &T::NoTape,
) -> Tensor0D<T::Tape> {
    mean(crate::tensor_ops::utils::binary_map(
        logits,
        target_probs,
        |logit, prob| logit.max(0.0) - logit * prob + (1.0 + (-logit.abs()).exp()).ln(),
        |logit, prob| 1.0 - prob - (1.0 + logit.exp()).recip(),
        |logit, _| -logit,
    ))
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
        let g = backward(loss);
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
        let g = backward(loss);
        assert_eq!(g.ref_gradient(&x), &[0.2, 0.2, -0.2, -0.2, 0.2]);
    }

    #[test]
    fn test_soft_cross_entropy() {
        let x = Tensor1D::new([-0.5722721, 0.8469643, 1.2063414, -1.0964301, 1.1945194]);
        let y = Tensor1D::new([0.10473672, 0.24449949, 0.3266706, 0.22253996, 0.10155323]);
        let loss = cross_entropy_with_logits_loss(x.trace(), &y);
        assert_eq!(loss.data(), &1.8713832);
        let g = backward(loss);
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
        let gradients = backward(loss);
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
        let logit = Tensor2D::new([
            [-0.4092005, -0.6706018, 0.9201696],
            [-1.6583557, 1.6978683, -1.4827578],
            [-0.9571696, -1.0971526, 0.8801755],
        ]);
        let prob = Tensor2D::new([
            [0.365251, 0.8322099, 0.482717],
            [0.168392, 0.7987092, 0.1177533],
            [0.7026833, 0.5563793, 0.6429267],
        ]);
        let loss = binary_cross_entropy_with_logits_loss(logit.trace(), &prob);
        assert_eq!(loss.data(), &0.70457286);

        let gradients = backward(loss);

        assert_eq!(
            gradients.ref_gradient(&logit),
            &[
                [0.003761424, -0.054871976, 0.025817735],
                [-0.0009343492, 0.0051718787, 0.0074731046],
                [-0.047248676, -0.03401173, 0.0071035423]
            ]
        );

        assert_eq!(
            gradients.ref_gradient(&prob),
            &[
                [0.04546672, 0.07451131, -0.10224107],
                [0.18426175, -0.18865204, 0.16475087],
                [0.10635218, 0.12190584, -0.097797275]
            ]
        );
    }

    #[test]
    fn test_bce_wide_range() {
        let logit = Tensor2D::new([[100.0; 3], [-100.0; 3], [-1.0, 0.0, 1.0]]);
        let targ = Tensor2D::new([[0.0, 0.5, 1.0]; 3]);

        let loss = binary_cross_entropy_with_logits_loss(logit.trace(), &targ);
        assert_eq!(loss.data(), &33.479965);

        let gradients = backward(loss);

        assert_eq!(
            gradients.ref_gradient(&logit),
            &[
                [0.11111111, 0.055555556, 0.0],
                [0.0, -0.055555556, -0.11111111],
                [0.029882379, 0.0, -0.02988238]
            ]
        );

        assert_eq!(
            gradients.ref_gradient(&targ),
            &[
                [-11.111112, -11.111112, -11.111112],
                [11.111112, 11.111112, 11.111112],
                [0.11111111, 0.0, -0.11111111]
            ]
        );
    }

    #[test]
    fn test_huber_loss() {
        let x = Tensor2D::new([
            [1.0095837, -1.0026205, -0.1126093, -0.1539351, -0.3688708],
            [2.6373475, 0.6761999, -1.3586733, 0.486154, -0.6206786],
            [-1.2967702, -0.1273358, 1.3558478, 0.0787393, 1.0921133],
        ]);
        let y = Tensor2D::new([
            [1.2569424, -1.2246597, 0.7995769, 0.0339246, -0.3688708],
            [1.472675, 0.8260061, 0.7839395, -0.0541475, -0.6206786],
            [-2.0449343, 1.8117315, 1.7505344, -1.2522424, 1.0921133],
        ]);

        let loss = huber_loss(x.trace(), &y, 0.5);
        assert_eq!(loss.data(), &0.24506615);

        let gradients = backward(loss);
        assert_eq!(
            gradients.ref_gradient(&x),
            &[
                [-0.016490579, 0.014802615, -0.033333335, -0.012523981, 0.0],
                [0.033333335, -0.0099870805, -0.033333335, 0.033333335, 0.0],
                [0.033333335, -0.033333335, -0.02631244, 0.033333335, 0.0]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&y),
            &[
                [0.016490579, -0.014802615, 0.033333335, 0.012523981, 0.0],
                [-0.033333335, 0.0099870805, 0.033333335, -0.033333335, 0.0],
                [-0.033333335, 0.033333335, 0.02631244, -0.033333335, 0.0]
            ]
        );
    }

    #[test]
    fn test_smooth_l1_loss() {
        let x = Tensor2D::new([
            [1.0095837, -1.0026205, -0.1126093, -0.1539351, -0.3688708],
            [2.6373475, 0.6761999, -1.3586733, 0.486154, -0.6206786],
            [-1.2967702, -0.1273358, 1.3558478, 0.0787393, 1.0921133],
        ]);
        let y = Tensor2D::new([
            [1.2569424, -1.2246597, 0.7995769, 0.0339246, -0.3688708],
            [1.472675, 0.8260061, 0.7839395, -0.0541475, -0.6206786],
            [-2.0449343, 1.8117315, 1.7505344, -1.2522424, 1.0921133],
        ]);

        let loss = smooth_l1_loss(x.trace(), &y, 0.5);
        assert_eq!(loss.data(), &0.4901323);

        let gradients = backward(loss);
        assert_eq!(
            gradients.ref_gradient(&x),
            &[
                [-0.032981157, 0.02960523, -0.06666667, -0.025047962, 0.0],
                [0.06666667, -0.019974161, -0.06666667, 0.06666667, 0.0],
                [0.06666667, -0.06666667, -0.05262488, 0.06666667, 0.0]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&y),
            &[
                [0.032981157, -0.02960523, 0.06666667, 0.025047962, 0.0],
                [-0.06666667, 0.019974161, 0.06666667, -0.06666667, 0.0],
                [-0.06666667, 0.06666667, 0.05262488, -0.06666667, 0.0]
            ]
        );
    }
}
