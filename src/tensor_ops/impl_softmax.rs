use crate::devices::{Device, DeviceReduce, MaxAccum, SubAccum};
use crate::gradients::Tape;
use crate::prelude::*;

/// Computes the [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) function across
/// `Axes`
///
/// **Pytorch equivalent**: `t.exp().sum(Axes).log()`
///
/// **Related functions**: [ln()], [sum()], [exp()], [log_softmax()], [softmax()]
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 4, 6> = TensorCreator::zeros();
/// let _: Tensor2D<2, 4> = t.logsumexp();
/// ```
///
/// Multi axis logsumexp:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 4, 6> = TensorCreator::zeros();
/// let _: Tensor1D<4> = t.logsumexp();
/// ```
pub fn logsumexp<T: Reduce<Axes>, Axes>(mut t: T) -> T::Reduced {
    let max = T::DeviceR::reduce::<MaxAccum>(t.data());
    T::DeviceR::broadcast_into_no_reset::<SubAccum>(t.mut_data(), max.as_ref());
    let mut result = ln(sum(exp(t)));
    <T::Reduced as HasDevice>::Device::add(result.mut_data(), max.as_ref());
    result
}

/// `log(softmax(t))` in numerically stable way across `Axes`. Does `t - logsumexp(t)` under the hood.
///
/// **Pytorch equivalent**: `t.log_softmax(Axes)`
///
/// **Related functions**: [logsumexp()], [softmax()]
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.log_softmax::<Axis<2>>();
/// ```
///
/// Using multi axis log_softmax:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.log_softmax::<Axes2<0, 2>>();
/// ```
pub fn log_softmax<T: Reduce<Axes>, Axes>(t: T) -> T {
    let (t, tape) = t.split_tape();
    let (lse, tape) = logsumexp(t.duplicate().put_tape(tape))
        .broadcast()
        .split_tape();
    sub(t.put_tape(tape), &lse)
}

/// Computes the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) across
/// `Axes`.
///
/// Equivalent to `exp(log_softmax(t))`.
///
/// **Pytorch equivalent**: `t.softmax(Axes)`
///
/// **Related functions**: [logsumexp()], [log_softmax()]
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.softmax::<Axis<2>>();
/// ```
///
/// Using multi axis softmax:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.softmax::<Axes2<1, 2>>();
/// ```
pub fn softmax<T: Reduce<Axes>, Axes>(t: T) -> T {
    exp(log_softmax(t))
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [logsumexp()] on `self` with `Axes`.
    pub fn logsumexp<T, Axes>(self) -> T where Self: ReduceTo<T, Axes> {
        logsumexp(self)
    }
    /// Calls [log_softmax()] on `self` with `Axes`
    pub fn log_softmax<Axes>(self) -> Self where Self: Reduce<Axes> {
        log_softmax(self)
    }
    /// Calls [softmax()] on `self` with `Axes`
    pub fn softmax<Axes>(self) -> Self where Self: Reduce<Axes> {
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
    use crate::tests::assert_close;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_logsumexp_0d() {
        let a = tensor(0.0);
        let r = a.trace().logsumexp();
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &1.0);
    }

    #[test]
    fn test_log_softmax_0d() {
        let a = tensor(0.0);
        let r = a.trace().log_softmax();
        assert_eq!(r.data(), &0.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &0.0);
    }

    #[test]
    fn test_softmax_0d() {
        let a = tensor(0.0);
        let r = a.trace().softmax();
        assert_eq!(r.data(), &1.0);
        let gradients = r.backward();
        assert_eq!(gradients.ref_gradient(&a), &0.0);
    }

    #[test]
    fn test_logsumexp_1d() {
        let a: Tensor1D<5> = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
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
        let a: Tensor1D<5> = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().log_softmax();
        assert_eq!(
            r.data(),
            &[-4.4519143, -3.4519143, -2.4519143, -1.4519143, -0.4519143]
        );
        let gradients = r.mean().backward();
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
        let a: Tensor1D<5> = tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().softmax();
        assert_eq!(
            r.data(),
            &[0.011656232, 0.031684924, 0.086128555, 0.23412168, 0.6364087]
        );
        let l = mul(r, &tensor([0.0, 0.0, 1.0, 0.0, 0.0]));
        assert_eq!(l.data(), &[0.0, 0.0, 0.086128555, 0.0, 0.0]);
        let gradients = l.mean().backward();
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
        let a: Tensor2D<2, 3> = tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r: Tensor1D<2, OwnedTape> = a.trace().logsumexp();
        assert_eq!(r.data(), &[0.40760595, 7.0509458]);
        let gradients = r.mean().backward();
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
        let a: Tensor2D<2, 3> = tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().log_softmax::<Axis<1>>();
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
        let a: Tensor2D<2, 3> = tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().softmax::<Axis<1>>();
        assert_eq!(
            r.data(),
            &[
                [0.09003058, 0.24472849, 0.66524094],
                [0.002355633, 0.047314156, 0.9503302]
            ]
        );
        let l = mul(r, &tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]));
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

    #[test]
    fn test_softmax_2d_0th_axis() {
        let a: Tensor2D<2, 3> = tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().softmax::<Axis<0>>();
        assert_eq!(
            r.data(),
            &[
                [0.047425874, 0.0066928514, 0.0009110514],
                [0.95257413, 0.9933072, 0.9990892]
            ]
        );
        let l = mul(r, &tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]));
        assert_eq!(l.data(), &[[0.047425874, 0.0, 0.0], [0.0, 0.9933072, 0.0]]);
        let gradients = backward(l.mean());
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.0075294436, -0.0011080095, 0.0],
                [-0.0075294436, 0.0011080056, 0.0]
            ]
        );
    }

    #[test]
    fn test_softmax_3d_to_1d_12() {
        let mut rng = StdRng::seed_from_u64(0);
        let t: Tensor3D<2, 3, 4> = TensorCreator::randn(&mut rng);
        let r = softmax::<_, Axes2<1, 2>>(t.trace());
        #[rustfmt::skip]
        assert_close(
            r.data(),
            &[
                [[0.08535644, 0.0987266, 0.00366116, 0.04927256], [0.01169326, 0.1515922, 0.00951258, 0.07721686], [0.0776206, 0.23813945, 0.19471556, 0.00249278]],
                [[0.01881982, 0.25171953, 0.02559674, 0.03725754], [0.04064152, 0.314442, 0.02427996, 0.04708378], [0.02791536, 0.14462142, 0.02221143, 0.04541067]],
            ],
        );
    }
}
