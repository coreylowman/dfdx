use super::device::StridedArray;
use crate::arrays::{BroadcastStridesTo, Dtype, Shape, StridesFor};
use std::sync::Arc;
use std::vec::Vec;

struct NdIndex<S: Shape> {
    indices: S::Concrete,
    shape: S::Concrete,
    strides: S::Concrete,
    next: Option<usize>,
}

impl<S: Shape> NdIndex<S> {
    fn new(shape: S, strides: StridesFor<S>) -> Self {
        let indices: S::Concrete = Default::default();
        let i: usize = strides
            .0
            .into_iter()
            .zip(indices.into_iter())
            .map(|(a, b)| a * b)
            .sum();
        Self {
            indices,
            shape: shape.concrete(),
            strides: strides.0,
            next: Some(i),
        }
    }
}

impl<S: Shape> NdIndex<S> {
    #[inline(always)]
    fn get_with_idx(&mut self) -> Option<(usize, S::Concrete)> {
        match self.next.as_mut() {
            Some(i) => {
                let idx = (*i, self.indices);
                if S::NUM_DIMS == 0 {
                    self.next = None;
                } else {
                    let mut curr = S::NUM_DIMS - 1;
                    loop {
                        self.indices[curr] += 1;
                        if self.indices[curr] < self.shape[curr] {
                            *i += self.strides[curr];
                            break;
                        } else {
                            *i -= (self.shape[curr] - 1) * self.strides[curr];
                            self.indices[curr] = 0;
                        }
                        if curr > 0 {
                            curr -= 1;
                        } else {
                            self.next = None;
                            break;
                        }
                    }
                }
                Some(idx)
            }
            None => None,
        }
    }
}

pub(crate) struct StridedRefIter<'a, S: Shape, Elem> {
    data: &'a Vec<Elem>,
    index: NdIndex<S>,
}

pub(crate) struct StridedMutIter<'a, S: Shape, Elem> {
    data: &'a mut Vec<Elem>,
    index: NdIndex<S>,
}

pub(crate) struct StridedRefIndexIter<'a, S: Shape, Elem> {
    data: &'a Vec<Elem>,
    index: NdIndex<S>,
}

pub(crate) struct StridedMutIndexIter<'a, S: Shape, Elem> {
    data: &'a mut Vec<Elem>,
    index: NdIndex<S>,
}

impl<S: Shape, Elem: Dtype> StridedArray<S, Elem> {
    pub(crate) fn buf_iter(&self) -> std::slice::Iter<'_, Elem> {
        self.data.iter()
    }

    pub(crate) fn buf_iter_mut(&mut self) -> std::slice::IterMut<'_, Elem> {
        std::sync::Arc::make_mut(&mut self.data).iter_mut()
    }

    pub(crate) fn iter(&self) -> StridedRefIter<S, Elem> {
        StridedRefIter {
            data: self.data.as_ref(),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    pub(crate) fn iter_mut(&mut self) -> StridedMutIter<S, Elem> {
        StridedMutIter {
            data: std::sync::Arc::make_mut(&mut self.data),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    pub(crate) fn iter_with_index(&self) -> StridedRefIndexIter<S, Elem> {
        StridedRefIndexIter {
            data: self.data.as_ref(),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    pub(crate) fn iter_mut_with_index(&mut self) -> StridedMutIndexIter<S, Elem> {
        StridedMutIndexIter {
            data: std::sync::Arc::make_mut(&mut self.data),
            index: NdIndex::new(self.shape, self.strides),
        }
    }
}

impl<S: Shape, Elem: Dtype> StridedArray<S, Elem> {
    pub(crate) fn iter_as<Axes, Dst: Shape>(&self, dst: &Dst) -> StridedRefIter<Dst, Elem>
    where
        S: BroadcastStridesTo<Dst, Axes>,
    {
        StridedRefIter {
            data: self.data.as_ref(),
            index: NdIndex::new(*dst, self.shape.broadcast_strides(self.strides)),
        }
    }

    pub(crate) fn iter_mut_as<Axes, Dst: Shape>(&mut self, dst: &Dst) -> StridedMutIter<Dst, Elem>
    where
        S: BroadcastStridesTo<Dst, Axes>,
    {
        StridedMutIter {
            data: Arc::make_mut(&mut self.data),
            index: NdIndex::new(*dst, self.shape.broadcast_strides(self.strides)),
        }
    }
}

pub(crate) trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
    fn next(&'_ mut self) -> Option<Self::Item<'_>>;
}

impl<'q, S: Shape, E> LendingIterator for StridedRefIter<'q, S, E> {
    type Item<'a> = &'a E where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index.get_with_idx().map(|(i, _)| &self.data[i])
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedMutIter<'q, S, E> {
    type Item<'a> = &'a mut E where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index.get_with_idx().map(|(i, _)| &mut self.data[i])
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedRefIndexIter<'q, S, E> {
    type Item<'a> = (&'a E, S::Concrete) where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index
            .get_with_idx()
            .map(|(i, idx)| (&self.data[i], idx))
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedMutIndexIter<'q, S, E> {
    type Item<'a> = (&'a mut E, S::Concrete) where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index
            .get_with_idx()
            .map(|(i, idx)| (&mut self.data[i], idx))
    }
}

#[cfg(test)]
mod tests {
    use crate::arrays::{Rank0, Rank1, Rank2, Rank3, StridesFor};

    use super::*;
    use std::vec;

    #[test]
    fn test_0d_contiguous_iter() {
        let s: StridedArray<Rank0, f32> = StridedArray {
            data: Arc::new(vec![0.0]),
            shape: (),
            strides: ().strides(),
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&0.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_1d_contiguous_iter() {
        let shape = Default::default();
        let s: StridedArray<Rank1<3>, f32> = StridedArray {
            data: Arc::new(vec![0.0, 1.0, 2.0]),
            shape,
            strides: shape.strides(),
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&0.0));
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&2.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_contiguous_iter() {
        let shape = Default::default();
        let s: StridedArray<Rank2<2, 3>, f32> = StridedArray {
            data: Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape,
            strides: shape.strides(),
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&2.0));
        assert_eq!(i.next(), Some(&3.0));
        assert_eq!(i.next(), Some(&4.0));
        assert_eq!(i.next(), Some(&5.0));
        assert_eq!(i.next(), Some(&6.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_broadcasted_0_iter() {
        let base: Rank1<3> = Default::default();
        let strides: StridesFor<Rank2<2, 3>> = base.broadcast_strides(base.strides());
        let s: StridedArray<Rank2<2, 3>, f32> = StridedArray {
            data: Arc::new(vec![1.0, 0.0, -1.0]),
            shape: Default::default(),
            strides,
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&0.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&0.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_broadcasted_1_iter() {
        let base: Rank1<2> = Default::default();
        let strides: StridesFor<Rank2<2, 3>> = base.broadcast_strides(base.strides());
        let s: StridedArray<Rank2<2, 3>, f32> = StridedArray {
            data: Arc::new(vec![1.0, -1.0]),
            shape: Default::default(),
            strides,
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_permuted_iter() {
        let s: StridedArray<Rank2<3, 2>, f32> = StridedArray {
            data: Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape: Default::default(),
            strides: StridesFor([1, 3]),
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&4.0));
        assert_eq!(i.next(), Some(&2.0));
        assert_eq!(i.next(), Some(&5.0));
        assert_eq!(i.next(), Some(&3.0));
        assert_eq!(i.next(), Some(&6.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_3d_broadcasted_iter() {
        let s: StridedArray<Rank3<3, 1, 2>, f32> = StridedArray {
            data: Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape: Default::default(),
            strides: StridesFor([2, 0, 1]),
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&2.0));
        assert_eq!(i.next(), Some(&3.0));
        assert_eq!(i.next(), Some(&4.0));
        assert_eq!(i.next(), Some(&5.0));
        assert_eq!(i.next(), Some(&6.0));
        assert!(i.next().is_none());
    }
}
