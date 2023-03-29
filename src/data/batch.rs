use crate::shapes::{Const, Dim};

use std::vec::Vec;

pub struct ExactBatcher<Size, I> {
    size: Size,
    iter: I,
}

pub struct Batcher<I> {
    size: usize,
    iter: I,
}

impl<const N: usize, I: Iterator> Iterator for ExactBatcher<Const<N>, I> {
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

impl<I: Iterator> Iterator for ExactBatcher<usize, I> {
    type Item = Vec<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.size);
        for _ in 0..self.size {
            batch.push(self.iter.next()?);
        }
        Some(batch)
    }
}

impl<I: Iterator> Iterator for Batcher<I> {
    type Item = Vec<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.size);

        for _ in 0..self.size {
            if let Some(item) = self.iter.next() {
                batch.push(item);
            } else {
                break;
            }
        }
        (!batch.is_empty()).then_some(batch)
    }
}

impl<Batch: Dim, I: ExactSizeIterator> ExactSizeIterator for ExactBatcher<Batch, I>
where
    Self: Iterator,
{
    fn len(&self) -> usize {
        self.iter.len() / self.size.size()
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Batcher<I>
where
    Self: Iterator,
{
    fn len(&self) -> usize {
        (self.iter.len() + self.size.size() - 1) / self.size.size()
    }
}

/// Create batches of items from an [Iterator]
pub trait IteratorBatchExt: Iterator {
    /// Return an [Iterator] where the items are either:
    /// - `[Self::Item; N]`, if `Size` is [`Const<N>`]
    /// - `Vec<Self::Item>`, if `Size` is [usize].
    ///
    /// **If the last batch contains fewer than `size` items, it is not returned.** To include this
    /// batch, use [IteratorBatchExt::batch_with_last].
    ///
    /// Const batches:
    /// ```rust
    /// # use dfdx::{prelude::*, data::IteratorBatchExt};
    /// let items: Vec<[usize; 5]> = (0..12).batch_exact(Const::<5>).collect();
    /// assert_eq!(&items, &[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]);
    /// ```
    ///
    /// Runtime batches:
    /// ```rust
    /// # use dfdx::{prelude::*, data::IteratorBatchExt};
    /// let items: Vec<Vec<usize>> = (0..12).batch_exact(5).collect();
    /// assert_eq!(&items, &[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]);
    /// ```
    fn batch_exact<Size: Dim>(self, size: Size) -> ExactBatcher<Size, Self>
    where
        Self: Sized,
    {
        ExactBatcher { size, iter: self }
    }

    /// Returns an [Iterator] containing all data in the input iterator grouped into batches of
    /// maximum length `size`. All batches except the last contain exactly `size` elements, and all
    /// batches contain at least one element.
    ///
    /// Example:
    /// ```rust
    /// # use dfdx::{prelude::*, data::IteratorBatchExt};
    /// let items: Vec<Vec<usize>> = (0..12).batch_with_last(5).collect();
    /// assert_eq!(&items, &[vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9], vec![10, 11]]);
    /// ```
    fn batch_with_last(self, size: usize) -> Batcher<Self>
    where
        Self: Sized,
    {
        Batcher { size, iter: self }
    }

    /// Deprecated, use [IteratorBatchExt::batch_exact] instead.
    #[deprecated]
    fn batch<Size: Dim>(self, size: Size) -> ExactBatcher<Size, Self>
    where
        Self: Sized,
    {
        ExactBatcher { size, iter: self }
    }
}
impl<I: Iterator> IteratorBatchExt for I {}
