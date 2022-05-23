use crate::prelude::*;

pub trait HasMeanMethod: Tensor {
    fn mean(self) -> Tensor0D<Self::TapeHolder>;
}

impl<T: Tensor> HasMeanMethod for T {
    /// Sums all the values in `self` and divides by number of values.
    ///
    /// Returns a [Tensor0D] (i.e. one number).
    fn mean(self) -> Tensor0D<Self::TapeHolder> {
        let result = Tensor0D::<NoTape>::new(T::Device::mean(self.data()));
        let (mut t, mut tape_holder) = self.split_tape_holder();
        let _result = result.phantom();
        tape_holder.add_operation(move |tape| {
            let g: f32 = *tape.ref_gradient(&_result);
            T::Device::map_assign(t.mut_data(), |v| *v = g / T::Array::NUM_ELEMENTS as f32);
            T::Device::add_assign(tape.mut_gradient(&t), t.data());
        });
        result.with_tape_holder(tape_holder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_0d() {
        let t: Tensor0D = Tensor0D::new(3.0);
        let r = t.trace().mean();
        assert_eq!(r.data(), &3.0);
        let gradients = backward(r);
        assert_eq!(gradients.ref_gradient(&t), &1.0);
    }

    #[test]
    fn test_mean_1d() {
        let t: Tensor1D<3> = Tensor1D::new([1.0, 2.0, 3.0]);
        let r: Tensor0D<WithTape> = t.trace().mean();
        assert_eq!(r.data(), &2.0);
        let gradients = backward(r);
        assert_eq!(gradients.ref_gradient(&t), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_mean_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r: Tensor0D<WithTape> = t.trace().mean();
        assert_eq!(r.data(), &3.5);
        let gradients = backward(r);
        assert_eq!(gradients.ref_gradient(&t), &[[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_mean_3d() {
        let t: Tensor3D<4, 2, 3> = Tensor3D::ones();
        let r: Tensor0D<WithTape> = t.trace().mean();
        assert_eq!(r.data(), &1.0);
        let gradients = backward(r);
        assert_eq!(gradients.ref_gradient(&t), &[[[1.0 / 24.0; 3]; 2]; 4]);
    }
}
