use crate::tensor_ops::TryStack;

pub struct Stacker<I> {
    iter: I,
}

impl<I: Iterator> Iterator for Stacker<I>
where
    I::Item: TryStack,
{
    type Item = <I::Item as TryStack>::Stacked;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|i| i.stack())
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Stacker<I>
where
    Self: Iterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

/// Stack items of an [Iterator] where Self::Item impls [TryStack].
pub trait IteratorStackExt: Iterator {
    /// Stacks items - depends on implementation of [TryStack] by the items.
    ///
    /// Example implementations:
    /// ```rust
    /// # use dfdx_core::{data::IteratorStackExt, prelude::*};
    /// # let dev: Cpu = Default::default();
    /// let a: Tensor<Rank1<3>, f32, _> = dev.zeros();
    /// let data = [[a.clone(), a.clone(), a]];
    /// // we can call stack on each item in the iterator:
    /// let _: Vec<Tensor<Rank2<3, 3>, f32, _>> = data.into_iter().stack().collect();
    /// ```
    fn stack(self) -> Stacker<Self>
    where
        Self: Sized,
    {
        Stacker { iter: self }
    }
}
impl<I: Iterator> IteratorStackExt for I {}
