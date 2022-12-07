use super::shape::{Const, Dyn, Shape};

pub trait TryFromNumElements: Shape {
    fn try_from_num_elements(num_elements: usize) -> Option<Self>;
}

impl TryFromNumElements for () {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        if num_elements == 1 {
            Some(())
        } else {
            None
        }
    }
}

impl<const M: usize> TryFromNumElements for (Const<M>,) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        if num_elements == M {
            Some(Default::default())
        } else {
            None
        }
    }
}

impl TryFromNumElements for (Dyn,) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        Some((Dyn(num_elements),))
    }
}

impl<const M: usize, const N: usize> TryFromNumElements for (Const<M>, Const<N>) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        let shape: Self = Default::default();
        if shape.num_elements() == num_elements {
            Some(shape)
        } else {
            None
        }
    }
}

impl<const N: usize> TryFromNumElements for (Dyn, Const<N>) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        if num_elements % N == 0 {
            Some((Dyn(num_elements / N), Const))
        } else {
            None
        }
    }
}

impl<const M: usize> TryFromNumElements for (Const<M>, Dyn) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        if num_elements % M == 0 {
            Some((Const, Dyn(num_elements / M)))
        } else {
            None
        }
    }
}
