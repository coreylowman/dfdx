use crate::arrays::{Dim, Dtype, Shape, C};
use crate::devices::cpu::StridedArray;

#[derive(Copy, Clone)]
pub(super) struct View<S: Shape, E: Dtype> {
    pub(super) ptr: *const E,
    pub(super) shape: S,
    pub(super) strides: S::Concrete,
}

#[derive(Copy, Clone)]
pub(super) struct ViewMut<S: Shape, E: Dtype> {
    pub(super) ptr: *mut E,
    pub(super) shape: S,
    pub(super) strides: S::Concrete,
}

impl<S: Shape, E: Dtype> StridedArray<S, E> {
    pub(super) fn view(&self) -> View<S, E> {
        View {
            ptr: self.data.as_ptr(),
            shape: self.shape,
            strides: self.strides.0,
        }
    }

    pub(super) fn view_mut(&mut self) -> ViewMut<S, E> {
        ViewMut {
            ptr: std::sync::Arc::make_mut(&mut self.data).as_mut_ptr(),
            shape: self.shape,
            strides: self.strides.0,
        }
    }
}

impl<D1: Dim, E: Dtype> View<(D1,), E> {
    pub(super) fn br0(self) -> View<(C<1>, D1), E> {
        View {
            ptr: self.ptr,
            shape: (C, self.shape.0),
            strides: [0, self.strides[0]],
        }
    }
    pub(super) fn br1(self) -> View<(D1, C<1>), E> {
        View {
            ptr: self.ptr,
            shape: (self.shape.0, C),
            strides: [self.strides[0], 0],
        }
    }
}

impl<D1: Dim, E: Dtype> ViewMut<(D1,), E> {
    pub(super) fn br0(self) -> ViewMut<(C<1>, D1), E> {
        ViewMut {
            ptr: self.ptr,
            shape: (C, self.shape.0),
            strides: [0, self.strides[0]],
        }
    }
    pub(super) fn br1(self) -> ViewMut<(D1, C<1>), E> {
        ViewMut {
            ptr: self.ptr,
            shape: (self.shape.0, C),
            strides: [self.strides[0], 0],
        }
    }
}

impl<D1: Dim, D2: Dim, E: Dtype> View<(D1, D2), E> {
    pub(super) fn tr(self) -> View<(D2, D1), E> {
        View {
            ptr: self.ptr,
            shape: (self.shape.1, self.shape.0),
            strides: [self.strides[1], self.strides[0]],
        }
    }
}

impl<D1: Dim, E: Dtype> View<(D1,), E> {
    pub(super) fn idx(&self, index: usize) -> &E {
        unsafe { &*self.ptr.add(index * self.strides[0]) }
    }
}
impl<D1: Dim, E: Dtype> ViewMut<(D1,), E> {
    pub(super) fn idx(&mut self, index: usize) -> &mut E {
        unsafe { &mut *self.ptr.add(index * self.strides[0]) }
    }
}

impl<D1: Dim, D2: Dim, E: Dtype> View<(D1, D2), E> {
    pub(super) fn idx(&self, index: usize) -> View<(D2,), E> {
        View {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1,),
            strides: [self.strides[1]],
        }
    }
}
impl<D1: Dim, D2: Dim, E: Dtype> ViewMut<(D1, D2), E> {
    pub(super) fn idx(&self, index: usize) -> ViewMut<(D2,), E> {
        ViewMut {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1,),
            strides: [self.strides[1]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, E: Dtype> View<(D1, D2, D3), E> {
    pub(super) fn idx(&self, index: usize) -> View<(D2, D3), E> {
        View {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1, self.shape.2),
            strides: [self.strides[1], self.strides[2]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, E: Dtype> ViewMut<(D1, D2, D3), E> {
    pub(super) fn idx(&self, index: usize) -> ViewMut<(D2, D3), E> {
        ViewMut {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1, self.shape.2),
            strides: [self.strides[1], self.strides[2]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, E: Dtype> View<(D1, D2, D3, D4), E> {
    pub(super) fn idx(&self, index: usize) -> View<(D2, D3, D4), E> {
        View {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1, self.shape.2, self.shape.3),
            strides: [self.strides[1], self.strides[2], self.strides[3]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, E: Dtype> ViewMut<(D1, D2, D3, D4), E> {
    pub(super) fn idx(&self, index: usize) -> ViewMut<(D2, D3, D4), E> {
        ViewMut {
            ptr: unsafe { self.ptr.add(index * self.strides[0]) },
            shape: (self.shape.1, self.shape.2, self.shape.3),
            strides: [self.strides[1], self.strides[2], self.strides[3]],
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, E: Dtype> View<(D1, D2, D3, D4, D5), E> {
    pub(super) fn idx(&self, index: usize) -> View<(D2, D3, D4, D5), E> {
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
    pub(super) fn idx(&self, index: usize) -> ViewMut<(D2, D3, D4, D5), E> {
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
