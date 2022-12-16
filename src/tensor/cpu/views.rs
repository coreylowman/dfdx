#![allow(dead_code)]

use super::StridedArray;
use crate::shapes::{Const, Dim, Shape};

#[derive(Clone, Copy)]
pub(crate) struct View<'a, S: Shape, E> {
    pub(crate) data: &'a [E],
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
}

impl<'a, S: Shape, E> View<'a, S, E> {
    #[inline(always)]
    pub(crate) fn new(data: &'a [E], shape: S) -> Self {
        Self {
            data,
            shape,
            strides: shape.strides(),
        }
    }

    #[inline(always)]
    pub(crate) fn ptr(&self) -> *const E {
        self.data.as_ptr()
    }
}

pub(crate) struct ViewMut<'a, S: Shape, E> {
    pub(crate) data: &'a mut [E],
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
}

impl<'a, S: Shape, E> ViewMut<'a, S, E> {
    #[inline(always)]
    pub(crate) fn new(data: &'a mut [E], shape: S) -> Self {
        Self {
            data,
            shape,
            strides: shape.strides(),
        }
    }

    #[inline(always)]
    pub(crate) fn ptr_mut(&mut self) -> *mut E {
        self.data.as_mut_ptr()
    }
}

impl<S: Shape, E> StridedArray<S, E> {
    #[inline(always)]
    pub(crate) fn view(&self) -> View<S, E> {
        View {
            data: self.data.as_slice(),
            shape: self.shape,
            strides: self.strides,
        }
    }
}

impl<S: Shape, E: Clone> StridedArray<S, E> {
    #[inline(always)]
    pub(crate) fn view_mut(&mut self) -> ViewMut<S, E> {
        ViewMut {
            data: std::sync::Arc::make_mut(&mut self.data).as_mut_slice(),
            shape: self.shape,
            strides: self.strides,
        }
    }
}

impl<'a, D1: Dim, E> View<'a, (D1,), E> {
    #[inline(always)]
    pub(crate) fn br0(self) -> View<'a, (Const<1>, D1), E> {
        View {
            data: self.data,
            shape: (Const, self.shape.0),
            strides: [0, self.strides[0]],
        }
    }
    #[inline(always)]
    pub(crate) fn br1(self) -> View<'a, (D1, Const<1>), E> {
        View {
            data: self.data,
            shape: (self.shape.0, Const),
            strides: [self.strides[0], 0],
        }
    }
}

impl<'a, D1: Dim, E> ViewMut<'a, (D1,), E> {
    #[inline(always)]
    pub(crate) fn br0(self) -> ViewMut<'a, (Const<1>, D1), E> {
        ViewMut {
            data: self.data,
            shape: (Const, self.shape.0),
            strides: [0, self.strides[0]],
        }
    }
    #[inline(always)]
    pub(crate) fn br1(self) -> ViewMut<'a, (D1, Const<1>), E> {
        ViewMut {
            data: self.data,
            shape: (self.shape.0, Const),
            strides: [self.strides[0], 0],
        }
    }
}

impl<'a, D1: Dim, D2: Dim, E> View<'a, (D1, D2), E> {
    #[inline(always)]
    pub(crate) fn tr(self) -> View<'a, (D2, D1), E> {
        View {
            data: self.data,
            shape: (self.shape.1, self.shape.0),
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

impl<'a, D1: Dim, E> View<'a, (D1,), E> {
    #[inline(always)]
    pub(crate) fn idx(&'a self, index: usize) -> &'a E {
        &self.data[index * self.strides[0]]
    }
}
impl<'a, D1: Dim, E> ViewMut<'a, (D1,), E> {
    #[inline(always)]
    pub(crate) fn idx_mut(&'a mut self, index: usize) -> &'a mut E {
        &mut self.data[index * self.strides[0]]
    }
}

impl<'a, D1: Dim, D2: Dim, E> View<'a, (D1, D2), E> {
    #[inline(always)]
    pub(crate) fn idx<'b>(&'b self, index: usize) -> View<'b, (D2,), E>
    where
        'a: 'b,
    {
        View {
            data: self.data.split_at(index * self.strides[0]).1,
            shape: (self.shape.1,),
            strides: [self.strides[1]],
        }
    }
}

impl<'a, D1: Dim, D2: Dim, E> ViewMut<'a, (D1, D2), E> {
    #[inline(always)]
    pub(crate) fn idx_mut<'b>(&'b mut self, index: usize) -> ViewMut<'b, (D2,), E>
    where
        'a: 'b,
    {
        ViewMut {
            data: self.data.split_at_mut(index * self.strides[0]).1,
            shape: (self.shape.1,),
            strides: [self.strides[1]],
        }
    }
}

impl<'a, D1: Dim, D2: Dim, D3: Dim, E> View<'a, (D1, D2, D3), E> {
    #[inline(always)]
    pub(crate) fn idx<'b>(&'b self, index: usize) -> View<'b, (D2, D3), E>
    where
        'a: 'b,
    {
        View {
            data: self.data.split_at(index * self.strides[0]).1,
            shape: (self.shape.1, self.shape.2),
            strides: [self.strides[1], self.strides[2]],
        }
    }
}

impl<'a, D1: Dim, D2: Dim, D3: Dim, E> ViewMut<'a, (D1, D2, D3), E> {
    #[inline(always)]
    pub(crate) fn idx_mut<'b>(&'b mut self, index: usize) -> ViewMut<'b, (D2, D3), E>
    where
        'a: 'b,
    {
        ViewMut {
            data: self.data.split_at_mut(index * self.strides[0]).1,
            shape: (self.shape.1, self.shape.2),
            strides: [self.strides[1], self.strides[2]],
        }
    }
}

impl<'a, D1: Dim, D2: Dim, D3: Dim, D4: Dim, E> View<'a, (D1, D2, D3, D4), E> {
    #[inline(always)]
    pub(crate) fn idx<'b>(&'b self, index: usize) -> View<'b, (D2, D3, D4), E>
    where
        'a: 'b,
    {
        View {
            data: self.data.split_at(index * self.strides[0]).1,
            shape: (self.shape.1, self.shape.2, self.shape.3),
            strides: [self.strides[1], self.strides[2], self.strides[3]],
        }
    }
}

impl<'a, D1: Dim, D2: Dim, D3: Dim, D4: Dim, E> ViewMut<'a, (D1, D2, D3, D4), E> {
    #[inline(always)]
    pub(crate) fn idx_mut<'b>(&'b mut self, index: usize) -> ViewMut<'b, (D2, D3, D4), E>
    where
        'a: 'b,
    {
        ViewMut {
            data: self.data.split_at_mut(index * self.strides[0]).1,
            shape: (self.shape.1, self.shape.2, self.shape.3),
            strides: [self.strides[1], self.strides[2], self.strides[3]],
        }
    }
}

impl<'a, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, E> View<'a, (D1, D2, D3, D4, D5), E> {
    #[inline(always)]
    pub(crate) fn idx<'b>(&'b self, index: usize) -> View<'b, (D2, D3, D4, D5), E>
    where
        'a: 'b,
    {
        View {
            data: self.data.split_at(index * self.strides[0]).1,
            shape: (self.shape.1, self.shape.2, self.shape.3, self.shape.4),
            strides: [
                self.strides[1],
                self.strides[2],
                self.strides[3],
                self.strides[4],
            ],
        }
    }
}

impl<'a, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, E> ViewMut<'a, (D1, D2, D3, D4, D5), E> {
    #[inline(always)]
    pub(crate) fn idx_mut<'b>(&'b mut self, index: usize) -> ViewMut<'b, (D2, D3, D4, D5), E>
    where
        'a: 'b,
    {
        ViewMut {
            data: self.data.split_at_mut(index * self.strides[0]).1,
            shape: (self.shape.1, self.shape.2, self.shape.3, self.shape.4),
            strides: [
                self.strides[1],
                self.strides[2],
                self.strides[3],
                self.strides[4],
            ],
        }
    }
}
