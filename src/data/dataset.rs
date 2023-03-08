use rand::prelude::{Rng, SliceRandom};
use std::vec::Vec;

/// A dataset with a known size that can be iterated over
/// in order or in shuffled order.
pub trait ExactSizeDataset {
    type Item<'a>
    where
        Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iter(&'_ self) -> Shuffled<'_, Self>
    where
        Self: Sized,
    {
        let indices: Vec<usize> = (0..self.len()).collect();
        Shuffled {
            data: self,
            indices,
        }
    }

    fn shuffled<R: Rng>(&'_ self, rng: &mut R) -> Shuffled<'_, Self>
    where
        Self: Sized,
    {
        let mut iter = self.iter();
        iter.indices.shuffle(rng);
        iter
    }
}

pub struct Shuffled<'a, D> {
    data: &'a D,
    indices: Vec<usize>,
}

impl<'a, D: ExactSizeDataset> Iterator for Shuffled<'a, D> {
    type Item = D::Item<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.pop().map(|i| self.data.get(i))
    }
}
impl<'a, D: ExactSizeDataset> ExactSizeIterator for Shuffled<'a, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }
}
