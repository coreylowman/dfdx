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

impl<'a, A, B, const N: usize> Collate for [&'a (A, B); N] {
    type Collated = ([&'a A; N], [&'a B; N]);
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

impl<'a, A, B> Collate for Vec<&'a (A, B)> {
    type Collated = (Vec<&'a A>, Vec<&'a B>);
    fn collated(self) -> Self::Collated {
        #[allow(clippy::map_identity)]
        self.into_iter().map(|(a, b)| (a, b)).unzip()
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
    /// Collates (unzips) items - very similar to [std::iter::Iterator::unzip],
    /// but works on items that are Vecs or arrays
    ///
    /// For example:
    /// - An item of `Vec<(usize, usize)>` becomes `(Vec<usize>, Vec<usize>)`
    /// - An item of `[(usize, usize); N]` becomes `([usize; N], [usize; N]`
    ///
    /// Example implementations:
    /// ```rust
    /// # use dfdx_core::data::IteratorCollateExt;
    /// let data = [[('a', 'b'); 10], [('c', 'd'); 10], [('e', 'f'); 10]];
    /// // we use collate to transform each batch:
    /// let mut iter = data.into_iter().collate();
    /// assert_eq!(iter.next().unwrap(), (['a'; 10], ['b'; 10]));
    /// assert_eq!(iter.next().unwrap(), (['c'; 10], ['d'; 10]));
    /// assert_eq!(iter.next().unwrap(), (['e'; 10], ['f'; 10]));
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

        let items = [&(1, 2), &(3, 4), &(5, 6)];
        assert_eq!(items.collated(), ([&1, &3, &5], [&2, &4, &6]));
    }

    #[test]
    fn test_collate_vec() {
        let items = std::vec![(1, 2), (3, 4), (5, 6)];
        let (a, b): (Vec<i32>, Vec<i32>) = items.collated();
        assert_eq!(a, [1, 3, 5]);
        assert_eq!(b, [2, 4, 6]);

        let items = std::vec![&(1, 2), &(3, 4), &(5, 6)];
        let (a, b): (Vec<&i32>, Vec<&i32>) = items.collated();
        assert_eq!(a, [&1, &3, &5]);
        assert_eq!(b, [&2, &4, &6]);
    }
}
