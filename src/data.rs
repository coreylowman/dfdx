//! A collection of data utility classes such as [Arange], [OneHotEncode], and [SubsetIterator].

use rand::prelude::SliceRandom;
use std::vec::Vec;

use crate::{
    shapes::{Const, Dyn, Rank1},
    tensor::{CopySlice, DeviceStorage, Tensor, ZerosTensor},
};

/// Generates a tensor with ordered data from 0 to `N`.
///
/// Examples:
/// ```rust
/// use dfdx::{prelude::*, data::Arange};
/// let dev: Cpu = Default::default();
/// let t = dev.arange::<5>();
/// assert_eq!(t.array(), [0.0, 1.0, 2.0, 3.0, 4.0]);
/// ```
pub trait Arange: DeviceStorage + ZerosTensor<f32> + CopySlice<f32> {
    fn arange<const N: usize>(&self) -> Tensor<Rank1<N>, f32, Self> {
        let mut data = Vec::with_capacity(N);
        for i in 0..N {
            data.push(i as f32);
        }
        let mut t = self.zeros();
        t.copy_from(&data);
        t
    }
}
impl<D: DeviceStorage + ZerosTensor<f32> + CopySlice<f32>> Arange for D {}

/// One hot encodes an array of class labels into a 2d tensor of probability
/// vectors. This can be used in tandem with [crate::losses::cross_entropy_with_logits_loss()].
///
/// Const Generic Arguments:
/// - `N` - the number of classes
///
/// Arguments:
/// - `class_labels` - an array of size `B` where each element is the class label
///
/// Outputs: [Tensor] with shape `(usize, Const<N>)`
///
/// Examples:
/// ```rust
/// use dfdx::{prelude::*, data::OneHotEncode};
/// let dev: Cpu = Default::default();
/// let class_labels = [0, 1, 2, 1, 1];
/// // NOTE: 5 is the batch size, 3 is the number of classes
/// let probs: Tensor<(Dyn<'B'>, Const<3>), f32, _> = dev.one_hot_encode::<3, 'B'>(&class_labels);
/// assert_eq!(&probs.as_vec(), &[
///     1.0, 0.0, 0.0,
///     0.0, 1.0, 0.0,
///     0.0, 0.0, 1.0,
///     0.0, 1.0, 0.0,
///     0.0, 1.0, 0.0,
/// ]);
/// ```
pub trait OneHotEncode: DeviceStorage + ZerosTensor<f32> + CopySlice<f32> {
    fn one_hot_encode<const N: usize, const B: char>(
        &self,
        labels: &[usize],
    ) -> Tensor<(Dyn<B>, Const<N>), f32, Self> {
        let mut data = Vec::with_capacity(labels.len() * N);
        for &l in labels {
            for i in 0..N {
                data.push(if i == l { 1.0 } else { 0.0 });
            }
        }
        let mut t = self.zeros_like(&(Dyn::<B>(labels.len()), Const::<N>));
        t.copy_from(&data);
        t
    }
}
impl<D: DeviceStorage + ZerosTensor<f32> + CopySlice<f32>> OneHotEncode for D {}

/// A utility class to simplify sampling a fixed number of indices for
/// data from a dataset.
///
/// Generic Arguments:
/// - `B` - The number of indices to sample for a batch.
///
/// Iterating a dataset in order:
/// ```rust
/// # use dfdx::{prelude::*, data::SubsetIterator};
/// let mut subsets = SubsetIterator::<5>::in_order(100);
/// assert_eq!(subsets.next(), Some([0, 1, 2, 3, 4]));
/// ```
///
/// Iterating a dataset in random order:
/// ```rust
/// # use dfdx::{prelude::*, data::SubsetIterator};
/// # use rand::prelude::*;
/// let mut rng = StdRng::seed_from_u64(0);
/// let mut subsets = SubsetIterator::<5>::shuffled(100, &mut rng);
/// assert_eq!(subsets.next(), Some([17, 4, 76, 81, 5]));
/// ```
pub struct SubsetIterator<const B: usize> {
    i: usize,
    indices: Vec<usize>,
}

impl<const B: usize> SubsetIterator<B> {
    pub fn in_order(n: usize) -> Self {
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            indices.push(i);
        }
        Self { i: 0, indices }
    }

    pub fn shuffled<R: rand::Rng>(n: usize, rng: &mut R) -> Self {
        let mut sampler = Self::in_order(n);
        sampler.indices.shuffle(rng);
        sampler
    }
}

impl<const B: usize> Iterator for SubsetIterator<B> {
    type Item = [usize; B];
    fn next(&mut self) -> Option<Self::Item> {
        if self.indices.len() < B || self.i + B > self.indices.len() {
            None
        } else {
            let mut batch = [0; B];
            batch.copy_from_slice(&self.indices[self.i..self.i + B]);
            self.i += B;
            Some(batch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_uses_all() {
        let mut seen: Vec<usize> = Vec::new();
        for batch in SubsetIterator::<5>::in_order(100) {
            seen.extend(batch.iter());
        }
        for i in 0..100 {
            assert!(seen.contains(&i));
        }
    }

    #[test]
    fn sampler_drops_last() {
        let mut seen: Vec<usize> = Vec::new();
        for batch in SubsetIterator::<6>::in_order(100) {
            seen.extend(batch.iter());
        }
        for i in 0..96 {
            assert!(seen.contains(&i));
        }
    }
}
