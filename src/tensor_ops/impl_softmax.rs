use crate::prelude::*;

/// Computes the [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) function.
/// Equivalent to `log(sum(exp(data)))` or `data.exp().sum(-1).log()`.
///
/// Calls [ln()], [sum_last_dim()], and [exp()]
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor0D::new(0.0);
/// let r = logsumexp(a);
/// assert_eq!(r.data(), &0.0);
/// ```
///
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
/// let r = a.logsumexp();
/// assert_eq!(r.data(), &2.4519143);
/// ```
///
/// See [log_softmax()] and [softmax()] for related functions.
pub fn logsumexp<T: Tensor<Dtype = f32>>(mut t: T) -> T::LastDimReduced {
    let max = T::Device::reduce_last_dim(t.data(), &mut f32::max);
    T::Device::sub_assign(t.mut_data(), max.as_ref());
    let mut result = ln(sum_last_dim(exp(t)));
    <T::LastDimReduced as HasDevice>::Device::add_assign(result.mut_data(), max.as_ref());
    result
}

/// Numerically stable computation of `log(softmax(t))`. Does `t - logsumexp(t)` under the hood.
///
/// See [logsumexp()] and [softmax()] for related functions
pub fn log_softmax<T: Tensor<Dtype = f32>>(t: T) -> T {
    let t_ = t.duplicate();
    let lse = logsumexp(t);
    let (lse, tape) = lse.split_tape();
    sub_broadcast_rhs_last(t_.put_tape(tape), lse)
}

/// Computes the [softmax](https://en.wikipedia.org/wiki/Softmax_function) function.
/// Equivalent to `t.log_softmax().exp()` or `exp(log_softmax(t))` or `exp(t) / sum(exp(t))`
///
/// See [logsumexp()] and [log_softmax()] for related functions.
pub fn softmax<T: Tensor<Dtype = f32>>(t: T) -> T {
    exp(log_softmax(t))
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [logsumexp()] on `self`.
    pub fn logsumexp(self) -> <Self as Tensor>::LastDimReduced {
        logsumexp(self)
    }

    /// Calls [log_softmax()] on `self`
    pub fn log_softmax(self) -> Self {
        log_softmax(self)
    }

    /// Calls [softmax()] on `self`
    pub fn softmax(self) -> Self {
        softmax(self)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp_0d() {
        let a = Tensor0D::new(0.0);
        let r = a.trace().logsumexp();
        assert_eq!(r.data(), &0.0);
        let gradients = backward(r);
        assert_eq!(gradients.ref_gradient(&a), &1.0);
    }

    #[test]
    fn test_log_softmax_0d() {
        let a = Tensor0D::new(0.0);
        let r = a.trace().log_softmax();
        assert_eq!(r.data(), &0.0);
        let gradients = backward(r);
        assert_eq!(gradients.ref_gradient(&a), &0.0);
    }

    #[test]
    fn test_softmax_0d() {
        let a = Tensor0D::new(0.0);
        let r = a.trace().softmax();
        assert_eq!(r.data(), &1.0);
        let gradients = backward(r);
        assert_eq!(gradients.ref_gradient(&a), &0.0);
    }

    #[test]
    fn test_logsumexp_1d() {
        let a: Tensor1D<5> = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().logsumexp();
        assert_eq!(r.data(), &2.4519143);
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[0.011656231, 0.03168492, 0.08612854, 0.23412165, 0.6364086]
        );
    }

    #[test]
    fn test_log_softmax_1d() {
        let a: Tensor1D<5> = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().log_softmax();
        assert_eq!(
            r.data(),
            &[-4.4519143, -3.4519143, -2.4519143, -1.4519143, -0.4519143]
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                0.18834378,
                0.16831508,
                0.11387146,
                -0.034121647,
                -0.43640864
            ]
        );
    }

    #[test]
    fn test_softmax_1d() {
        let a: Tensor1D<5> = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().softmax();
        assert_eq!(
            r.data(),
            &[0.011656232, 0.031684924, 0.086128555, 0.23412168, 0.6364087]
        );
        let l = mul(r, &Tensor1D::new([0.0, 0.0, 1.0, 0.0, 0.0]));
        assert_eq!(l.data(), &[0.0, 0.0, 0.086128555, 0.0, 0.0]);
        let gradients = backward(l.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                -0.00020078686,
                -0.00054579525,
                0.015742086,
                -0.0040329117,
                -0.010962591
            ]
        );
    }

    #[test]
    fn test_logsumexp_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r: Tensor1D<2, OwnsTape> = a.trace().logsumexp();
        assert_eq!(r.data(), &[0.40760595, 7.0509458]);
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.045015287, 0.12236424, 0.33262047],
                [0.0011778167, 0.023657078, 0.47516513]
            ]
        );
    }

    #[test]
    fn test_log_softmax_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().log_softmax();
        assert_eq!(
            r.data(),
            &[
                [-2.407606, -1.4076059, -0.40760595],
                [-6.0509458, -3.0509458, -0.05094576]
            ]
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.12165138, 0.044302434, -0.1659538],
                [0.16548885, 0.14300959, -0.30849844]
            ]
        );
    }

    #[test]
    fn test_softmax_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().softmax();
        assert_eq!(
            r.data(),
            &[
                [0.09003058, 0.24472849, 0.66524094],
                [0.002355633, 0.047314156, 0.9503302]
            ]
        );
        let l = mul(r, &Tensor2D::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]));
        assert_eq!(l.data(), &[[0.09003058, 0.0, 0.0], [0.0, 0.047314156, 0.0]]);
        let gradients = backward(l.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.01365418, -0.0036721744, -0.009982005],
                [-1.85758e-5, 0.0075125876, -0.0074940124]
            ]
        );
    }
}
