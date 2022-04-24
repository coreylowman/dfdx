use crate::prelude::*;

pub trait HasSoftmaxMethod: Tensor + HasSumLastMethod + Sized {
    fn logsumexp(self) -> <Self as HasSumLastMethod>::Output;
    fn log_softmax(self) -> Self;

    fn softmax(self) -> Self {
        self.log_softmax().exp()
    }
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*], $ax:expr) => {
impl<$(const $Vs: usize, )* H: TapeHolder> HasSoftmaxMethod for $typename<$($Vs, )* H> {
    fn logsumexp(mut self) -> <Self as HasSumLastMethod>::Output {
        let max = self.data().map_axis($ax, |arr| arr.iter().cloned().reduce(f32::max).unwrap()).insert_axis($ax);
        for mut sub_ax in self.mut_data().lanes_mut($ax) {
            let sub_ax_max = sub_ax.iter().cloned().reduce(f32::max).unwrap();
            sub_ax.map_inplace(|v| *v -= sub_ax_max);
        }
        let mut result = self.exp().sum_last().ln();
        *result.mut_data() = result.data() + max;
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

tensor_impl!(Tensor1D, [M], ndarray::Axis(0));
tensor_impl!(Tensor2D, [M, N], ndarray::Axis(1));
// tensor_impl!(Tensor3D, [M, N, O]);
// tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_logsumexp_1d() {
        let a: Tensor1D<5> = Tensor1D::new(arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]));
        let r = a.with_tape().logsumexp();
        assert_eq!(r.data(), arr1(&[2.4519143]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(a.id()).to_shape((5,)).unwrap(),
            arr1(&[0.011656231, 0.03168492, 0.08612854, 0.23412165, 0.6364086])
        );
    }

    #[test]
    fn test_log_softmax_1d() {
        let a: Tensor1D<5> = Tensor1D::new(arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]));
        let r = a.with_tape().log_softmax();
        assert_eq!(
            r.data(),
            arr1(&[-4.4519143, -3.4519143, -2.4519143, -1.4519143, -0.4519143])
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(a.id()).to_shape((5,)).unwrap(),
            arr1(&[
                0.18834378,
                0.16831508,
                0.11387146,
                -0.034121647,
                -0.43640864
            ])
        );
    }

    #[test]
    fn test_softmax_1d() {
        let a: Tensor1D<5> = Tensor1D::new(arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]));
        let r = a.with_tape().softmax();
        assert_eq!(
            r.data(),
            arr1(&[0.011656232, 0.031684924, 0.086128555, 0.23412168, 0.6364087])
        );
        let l = &Tensor1D::new(arr1(&[0.0, 0.0, 1.0, 0.0, 0.0])) * r;
        assert_eq!(l.data(), arr1(&[0.0, 0.0, 0.086128555, 0.0, 0.0]));
        let gradients = backward(l.mean());
        assert_eq!(
            gradients.gradient_for(a.id()).to_shape((5,)).unwrap(),
            arr1(&[
                -0.00020078686,
                -0.00054579525,
                0.015742086,
                -0.0040329117,
                -0.010962591
            ])
        );
    }

    #[test]
    fn test_logsumexp_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]));
        let r = a.with_tape().logsumexp();
        assert_eq!(r.data(), arr2(&[[0.40760595], [7.0509458]]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(a.id()).to_shape(a.shape()).unwrap(),
            arr2(&[
                [0.045015287, 0.12236424, 0.33262047],
                [0.0011778167, 0.023657078, 0.47516513]
            ])
        );
    }

    #[test]
    fn test_log_softmax_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]));
        let r = a.with_tape().log_softmax();
        assert_eq!(
            r.data(),
            arr2(&[
                [-2.407606, -1.4076059, -0.40760595],
                [-6.0509458, -3.0509458, -0.05094576]
            ])
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients.gradient_for(a.id()).to_shape(a.shape()).unwrap(),
            arr2(&[
                [0.12165138, 0.044302434, -0.1659538],
                [0.16548885, 0.14300959, -0.30849844]
            ])
        );
    }

    #[test]
    fn test_softmax_2d() {
        let a: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]));
        let r = a.with_tape().softmax();
        assert_eq!(
            r.data(),
            arr2(&[
                [0.09003058, 0.24472849, 0.66524094],
                [0.002355633, 0.047314156, 0.9503302]
            ])
        );
        let l = &Tensor2D::new(arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])) * r;
        assert_eq!(
            l.data(),
            arr2(&[[0.09003058, 0.0, 0.0], [0.0, 0.047314156, 0.0]])
        );
        let gradients = backward(l.mean());
        assert_eq!(
            gradients.gradient_for(a.id()).to_shape(a.shape()).unwrap(),
            arr2(&[
                [0.01365418, -0.0036721744, -0.009982005],
                [-1.85758e-5, 0.0075125876, -0.0074940124]
            ])
        );
    }
}
