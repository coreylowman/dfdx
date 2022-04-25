use super::ops::add_unary_op;
use crate::prelude::*;
use ndarray::prelude::*;

pub trait HasMaxLastMethod: Tensor {
    type Output: Tensor;
    fn max_last(self) -> Self::Output;
}

macro_rules! max_last_impl {
    ($typename:ident, [$($Vs:tt),*], [$($Zs:tt),*], $ax:expr) => {
impl<$(const $Vs: usize, )* H: TapeHolder> HasMaxLastMethod for $typename<$($Vs, )* H> {
    type Output = $typename<$($Zs, )* H>;
    fn max_last(self) -> Self::Output {
        let result = <$typename<$($Zs, )* H> as Tensor>::NoTape::new(self.data().map_axis($ax, |arr| arr.iter().cloned().reduce(f32::max).unwrap()).insert_axis($ax));
        let (t, mut tape_holder) = self.split_tape_holder();
        tape_holder.update_with(|tape| {
            let mut deriv = t.data().clone();
            for mut sub_ax in deriv.lanes_mut($ax) {
                let max = sub_ax.iter().cloned().reduce(f32::max).unwrap();
                sub_ax.map_inplace(|v| *v = if *v == max { 1.0 } else { 0.0 });
            }
            add_unary_op(tape, (&t, &result), deriv)
        });
        result.with_tape_holder(tape_holder)
    }
}
    };
}

max_last_impl!(Tensor1D, [M], [1], Axis(0));
max_last_impl!(Tensor2D, [M, N], [M, 1], Axis(1));
max_last_impl!(Tensor3D, [M, N, O], [M, N, 1], Axis(2));
max_last_impl!(Tensor4D, [M, N, O, P], [M, N, O, 1], Axis(3));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_last_1d() {
        let t: Tensor1D<3> = Tensor1D::new(arr1(&[1.0, 2.0, 3.0]));
        let r = t.with_tape().max_last();
        assert_eq!(r.data(), arr1(&[3.0]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients
                .gradient_for(t.id())
                .clone()
                .into_shape((3,))
                .unwrap(),
            arr1(&[0.0, 0.0, 1.0])
        );
    }

    #[test]
    fn test_max_last_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new(arr2(&[[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]]));
        let r = t.with_tape().max_last();
        assert_eq!(r.data(), arr2(&[[3.0], [6.0]]));
        let gradients = backward(r.mean());
        assert_eq!(
            gradients
                .gradient_for(t.id())
                .clone()
                .into_shape((2, 3))
                .unwrap(),
            arr2(&[[0.0, 0.0, 0.5], [0.5, 0.0, 0.0]])
        );
    }

    #[test]
    fn test_max_last_3d() {
        let t: Tensor3D<2, 2, 3> = Tensor3D::new(Array3::from(
            [
                [[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]],
                [[-1.0, -2.0, -3.0], [-6.0, -5.0, -4.0]],
            ]
            .to_vec(),
        ));
        let r = t.with_tape().max_last();
        assert_eq!(
            r.data(),
            Array3::from([[[3.0], [6.0]], [[-1.0], [-4.0]]].to_vec())
        );
        let gradients = backward(r.mean());
        assert_eq!(
            gradients
                .gradient_for(t.id())
                .clone()
                .into_shape((2, 2, 3))
                .unwrap(),
            Array3::from(
                [
                    [[0.0, 0.0, 0.25], [0.25, 0.0, 0.0]],
                    [[0.25, 0.0, 0.0], [0.0, 0.0, 0.25]]
                ]
                .to_vec()
            )
        );
    }
}
