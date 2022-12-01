#![allow(dead_code)]

use super::StridedArray;
use crate::arrays::{Const, Dim, Dtype, Shape};

#[derive(Copy, Clone)]
pub(crate) struct View<S: Shape, E: Dtype> {
    pub(crate) ptr: *const E,
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
}

impl<S: Shape, E: Dtype> View<S, E> {
    #[inline(always)]
    pub(crate) fn new(ptr: *const E, shape: S) -> Self {
        Self {
            ptr,
            shape,
            strides: shape.strides(),
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) struct ViewMut<S: Shape, E: Dtype> {
    pub(crate) ptr: *mut E,
    pub(crate) shape: S,
    pub(crate) strides: S::Concrete,
}

impl<S: Shape, E: Dtype> ViewMut<S, E> {
    #[inline(always)]
    pub(crate) fn new(ptr: *mut E, shape: S) -> Self {
        Self {
            ptr,
            shape,
            strides: shape.strides(),
        }
    }
}

impl<S: Shape, E: Dtype> StridedArray<S, E> {
    #[inline(always)]
    pub(crate) fn view(&self) -> View<S, E> {
        View {
            ptr: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides,
        }
    }

    #[inline(always)]
    pub(crate) fn view_mut(&mut self) -> ViewMut<S, E> {
        ViewMut {
            ptr: std::sync::Arc::make_mut(&mut self.data).as_mut_ptr(),
            shape: self.shape,
            strides: self.strides,
        }
    }
}

impl<D1: Dim, E: Dtype> View<(D1,), E> {
    #[inline(always)]
    pub(crate) fn br0(self) -> View<(Const<1>, D1), E> {
        View {
            ptr: self.ptr,
            shape: (Const, self.shape.0),
            strides: [0, self.strides[0]],
        }
    }
    #[inline(always)]
    pub(crate) fn br1(self) -> View<(D1, Const<1>), E> {
        View {
            ptr: self.ptr,
            shape: (self.shape.0, Const),
            strides: [self.strides[0], 0],
        }
    }
}

impl<D1: Dim, E: Dtype> ViewMut<(D1,), E> {
    #[inline(always)]
    pub(crate) fn br0(self) -> ViewMut<(Const<1>, D1), E> {
        ViewMut {
            ptr: self.ptr,
            shape: (Const, self.shape.0),
            strides: [0, self.strides[0]],
        }
    }
    #[inline(always)]
    pub(crate) fn br1(self) -> ViewMut<(D1, Const<1>), E> {
        ViewMut {
            ptr: self.ptr,
            shape: (self.shape.0, Const),
            strides: [self.strides[0], 0],
        }
    }
}

impl<D1: Dim, D2: Dim, E: Dtype> View<(D1, D2), E> {
    #[inline(always)]
    pub(crate) fn tr(self) -> View<(D2, D1), E> {
        View {
            ptr: self.ptr,
            shape: (self.shape.1, self.shape.0),
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

impl<D1: Dim, E: Dtype> View<(D1,), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> &E {
        unsafe { &*self.ptr.add(index * self.strides[0]) }
    }
}
impl<D1: Dim, E: Dtype> ViewMut<(D1,), E> {
    #[inline(always)]
    pub(crate) fn idx(&mut self, index: usize) -> &mut E {
        unsafe { &mut *self.ptr.add(index * self.strides[0]) }
    }
}

impl<D1: Dim, D2: Dim, E: Dtype> View<(D1, D2), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> View<(D2,), E> {
        View {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1,),
            strides: [self.strides[1]],
        }
    }
}
impl<D1: Dim, D2: Dim, E: Dtype> ViewMut<(D1, D2), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> ViewMut<(D2,), E> {
        ViewMut {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1,),
            strides: [self.strides[1]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, E: Dtype> View<(D1, D2, D3), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> View<(D2, D3), E> {
        View {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1, self.shape.2),
            strides: [self.strides[1], self.strides[2]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, E: Dtype> ViewMut<(D1, D2, D3), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> ViewMut<(D2, D3), E> {
        ViewMut {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1, self.shape.2),
            strides: [self.strides[1], self.strides[2]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, E: Dtype> View<(D1, D2, D3, D4), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> View<(D2, D3, D4), E> {
        View {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1, self.shape.2, self.shape.3),
            strides: [self.strides[1], self.strides[2], self.strides[3]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, E: Dtype> ViewMut<(D1, D2, D3, D4), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> ViewMut<(D2, D3, D4), E> {
        ViewMut {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1, self.shape.2, self.shape.3),
            strides: [self.strides[1], self.strides[2], self.strides[3]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, E: Dtype> View<(D1, D2, D3, D4, D5), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> View<(D2, D3, D4, D5), E> {
        View {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
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

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, E: Dtype> ViewMut<(D1, D2, D3, D4, D5), E> {
    #[inline(always)]
    pub(crate) fn idx(&self, index: usize) -> ViewMut<(D2, D3, D4, D5), E> {
        ViewMut {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
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
