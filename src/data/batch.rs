use crate::shapes::{Const, Dim};

use std::vec::Vec;

pub struct Batcher<Size, I> {
    size: Size,
    iter: I,
}

impl<const N: usize, I: Iterator> Iterator for Batcher<Const<N>, I> {
    type Item = [I::Item; N];
    fn next(&mut self) -> Option<Self::Item> {
        let items = [(); N].map(|_| self.iter.next());
        if items.iter().any(Option::is_none) {
            None
        } else {
            Some(items.map(Option::unwrap))
        }
    }
}

impl<I: Iterator> Iterator for Batcher<usize, I> {
    type Item = Vec<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.size);
        for _ in 0..self.size {
            batch.push(self.iter.next()?);
        }
        Some(batch)
    }
}

/// Create batches of items from an [Iterator]
pub trait IteratorBatchExt: Iterator {
    /// Return an [Iterator] where the items are either:
    /// - `[Self::Item; N]`, if `Size` is [`Const<N>`]
    /// - `Vec<Self::Item>`, if `Size` is [usize].
    ///
    /// **Drop last is not supported - always returns exact batches**
    ///
    /// Const batches:
    /// ```rust
    /// # use dfdx::{prelude::*, data::IteratorBatchExt};
    /// let items: Vec<[usize; 5]> = (0..12).batch(Const::<5>).collect();
    /// assert_eq!(&items, &[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]);
    /// ```
    ///
    /// Runtime batches:
    /// ```rust
    /// # use dfdx::{prelude::*, data::IteratorBatchExt};
    /// let items: Vec<Vec<usize>> = (0..12).batch(5).collect();
    /// assert_eq!(&items, &[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]);
    /// ```
    fn batch<Size: Dim>(self, size: Size) -> Batcher<Size, Self>
    where
        Self: Sized,
    {
        Batcher { size, iter: self }
    }
}
impl<I: Iterator> IteratorBatchExt for I {}
