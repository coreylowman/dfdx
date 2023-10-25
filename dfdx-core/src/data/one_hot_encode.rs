use std::vec::Vec;

use crate::{
    shapes::*,
    tensor::{Storage, Tensor, TensorFromVec, ZerosTensor},
};

/// One hot encodes an array of class labels into a 2d tensor of probability
/// vectors. This can be used in tandem with [crate::losses::cross_entropy_with_logits_loss()].
pub trait OneHotEncode<E: Dtype>: Storage<E> + ZerosTensor<E> + TensorFromVec<E> {
    /// One hot encodes an array or vec into a tensor.
    ///
    /// Arguments:
    /// - `n` - the numnber of classes to use to encode, can be `Const` or `usize`
    /// - `class_labels` - either an array `[usize; N]`, or `Vec<usize>`
    ///
    /// Const class labels and const n:
    /// ```rust
    /// # use dfdx_core::{prelude::*, data::OneHotEncode};
    /// # let dev: Cpu = Default::default();
    /// let class_labels = [0, 1, 2, 1, 1];
    /// let probs: Tensor<Rank2<5, 3>, f32, _> = dev.one_hot_encode(Const::<3>, class_labels);
    /// assert_eq!(probs.array(), [
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    /// ]);
    /// ```
    ///
    /// Runtime class labels and const n:
    /// ```rust
    /// # use dfdx_core::{prelude::*, data::OneHotEncode};
    /// # let dev: Cpu = Default::default();
    /// let class_labels = [0, 1, 2, 1, 1];
    /// let probs: Tensor<(Const<5>, usize), f32, _> = dev.one_hot_encode(3, class_labels);
    /// assert_eq!(&probs.as_vec(), &[
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 1.0, 0.0,
    /// ]);
    /// ```
    ///
    /// Const class labels and runtime n:
    /// ```rust
    /// # use dfdx_core::{prelude::*, data::OneHotEncode};
    /// # let dev: Cpu = Default::default();
    /// let class_labels = std::vec![0, 1, 2, 1, 1];
    /// let probs: Tensor<(usize, Const<3>), f32, _> = dev.one_hot_encode(Const, class_labels);
    /// assert_eq!(&probs.as_vec(), &[
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 1.0, 0.0,
    /// ]);
    /// ```
    ///
    /// Runtime both:
    /// ```rust
    /// # use dfdx_core::{prelude::*, data::OneHotEncode};
    /// # let dev: Cpu = Default::default();
    /// let class_labels = std::vec![0, 1, 2, 1, 1];
    /// let probs: Tensor<(usize, usize), f32, _> = dev.one_hot_encode(3, class_labels);
    /// assert_eq!(&probs.as_vec(), &[
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 1.0, 0.0,
    /// ]);
    /// ```
    fn one_hot_encode<Lbls: Array<usize>, N: Dim>(
        &self,
        n: N,
        labels: Lbls,
    ) -> Tensor<(Lbls::Dim, N), E, Self> {
        let l = labels.dim();
        let mut data = Vec::with_capacity(l.size() * n.size());
        for l in labels.into_iter() {
            for i in 0..n.size() {
                data.push(if i == l {
                    E::from_usize(1).unwrap()
                } else {
                    E::from_usize(0).unwrap()
                });
            }
        }
        self.tensor_from_vec(data, (l, n))
    }
}
impl<E: Dtype, D: Storage<E> + ZerosTensor<E> + TensorFromVec<E>> OneHotEncode<E> for D {}
