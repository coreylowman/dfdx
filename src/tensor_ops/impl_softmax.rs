use crate::prelude::*;

pub trait HasSoftmaxMethod: Tensor + HasSumLastMethod + Sized {
    fn logsumexp(self) -> <Self as HasSumLastMethod>::Output;
    fn log_softmax(self) -> Self;
    fn softmax(self) -> Self {
        self.log_softmax().exp()
    }
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: TapeHolder> HasSoftmaxMethod for $typename<$($Vs, )* H> {
    fn logsumexp(mut self) -> <Self as HasSumLastMethod>::Output {
        let max = self.data().reduce_inner(f32::max);
        self.mut_data().map_assign_inner(|inner| {
            let max = inner.reduce(f32::max);
            inner.map_assign_elems(|v| *v -= max);
        });
        let mut result = self.exp().sum_last().ln();
        result.mut_data().add_assign(&max);
        result
    }

    fn log_softmax(self) -> Self {
        let (x, x_) = self.duplicate();
        let (lse, tape_holder) = x.logsumexp().split_tape_holder();
        x_.with_tape_holder(tape_holder) - &lse
    }
}
    };
}

tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
// tensor_impl!(Tensor3D, [M, N, O]);
// tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp_1d() {
        let a: Tensor1D<5> = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.with_tape().logsumexp();
        assert_eq!(r.data(), &2.4519143);
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient(&a),
            &[0.011656231, 0.03168492, 0.08612854, 0.23412165, 0.6364086]
        );
    }

    #[test]
    fn test_log_softmax_1d() {
        let a: Tensor1D<5> = Tensor1D::new([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.with_tape().log_softmax();
        assert_eq!(
            r.data(),
            &[-4.4519143, -3.4519143, -2.4519143, -1.4519143, -0.4519143]
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient(&a),
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
        let r = a.with_tape().softmax();
        assert_eq!(
            r.data(),
            &[0.011656232, 0.031684924, 0.086128555, 0.23412168, 0.6364087]
        );
        let l = &Tensor1D::new([0.0, 0.0, 1.0, 0.0, 0.0]) * r;
        assert_eq!(l.data(), &[0.0, 0.0, 0.086128555, 0.0, 0.0]);
        let gradients = backward(l.mean());
        assert_eq!(
            gradients.gradient(&a),
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
        let r: Tensor1D<2, WithTape> = a.with_tape().logsumexp();
        assert_eq!(r.data(), &[0.40760595, 7.0509458]);
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient(&a),
            &[
                [0.045015287, 0.12236424, 0.33262047],
                [0.0011778167, 0.023657078, 0.47516513]
            ]
        );
    }

    #[test]
    fn test_log_softmax_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.with_tape().log_softmax();
        assert_eq!(
            r.data(),
            &[
                [-2.407606, -1.4076059, -0.40760595],
                [-6.0509458, -3.0509458, -0.05094576]
            ]
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient(&a),
            &[
                [0.12165138, 0.044302434, -0.1659538],
                [0.16548885, 0.14300959, -0.30849844]
            ]
        );
    }

    #[test]
    fn test_softmax_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.with_tape().softmax();
        assert_eq!(
            r.data(),
            &[
                [0.09003058, 0.24472849, 0.66524094],
                [0.002355633, 0.047314156, 0.9503302]
            ]
        );
        let l = &Tensor2D::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) * r;
        assert_eq!(l.data(), &[[0.09003058, 0.0, 0.0], [0.0, 0.047314156, 0.0]]);
        let gradients = backward(l.mean());
        assert_eq!(
            gradients.gradient(&a),
            &[
                [0.01365418, -0.0036721744, -0.009982005],
                [-1.85758e-5, 0.0075125876, -0.0074940124]
            ]
        );
    }
}
