use std::{mem::MaybeUninit, vec::Vec};

/// Collates `Self` into some other type.
/// Generally similar to an unzip method;
/// Transforms `[(A, B); N]` into `([A; N], [B; N])`;
/// Transforms `Vec<(A, B)>` into `(Vec<A>, Vec<B>)`.
pub trait Collate {
    type Collated;
    fn collated(self) -> Self::Collated;
}

impl<A, B, const N: usize> Collate for [(A, B); N] {
    type Collated = ([A; N], [B; N]);
    fn collated(self) -> Self::Collated {
        let mut a_n = [(); N].map(|_| MaybeUninit::uninit());
        let mut b_n = [(); N].map(|_| MaybeUninit::uninit());

        for (i, (a, b)) in self.into_iter().enumerate() {
            a_n[i].write(a);
            b_n[i].write(b);
        }

        let a_n = unsafe { a_n.map(|a| a.assume_init()) };
        let b_n = unsafe { b_n.map(|b| b.assume_init()) };

        (a_n, b_n)
    }
}

impl<A, B> Collate for Vec<(A, B)> {
    type Collated = (Vec<A>, Vec<B>);
    fn collated(self) -> Self::Collated {
        self.into_iter().unzip()
    }
}

pub struct Collator<I> {
    iter: I,
}

impl<I: Iterator> Iterator for Collator<I>
where
    I::Item: Collate,
{
    type Item = <I::Item as Collate>::Collated;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|i| i.collated())
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Collator<I>
where
    Self: Iterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

/// Collate an [Iterator] where Self::Item impls [Collate].
pub trait IteratorCollateExt: Iterator {
    /// Collates items - depends on implementation of [Collate] by the items.
    ///
    /// Example implementations:
    /// ```rust
    /// # use dfdx::data::IteratorCollateExt;
    /// // batches of data where the items in the batch are (i32, i32):
    /// let data = [[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10), (11, 12)]];
    /// // we use collate to transform each batch:
    /// let batches: Vec<([i32; 3], [i32; 3])> = data.into_iter().collate().collect();
    /// assert_eq!(
    ///     &batches,
    ///     &[
    ///         ([1, 3, 5], [2, 4, 6]),
    ///         ([7, 9, 11], [8, 10, 12]),
    ///     ],
    /// );
    /// ```
    fn collate(self) -> Collator<Self>
    where
        Self: Sized,
    {
        Collator { iter: self }
    }
}
impl<I: Iterator> IteratorCollateExt for I {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_collate_array() {
        let items = [(1, 2), (3, 4), (5, 6)];
        assert_eq!(items.collated(), ([1, 3, 5], [2, 4, 6]));
    }

    #[test]
    fn test_collate_vec() {
        let items = std::vec![(1, 2), (3, 4), (5, 6)];
        let (a, b): (Vec<i32>, Vec<i32>) = items.collated();
        assert_eq!(&a, &[1, 3, 5]);
        assert_eq!(&b, &[2, 4, 6]);
    }
}
