pub trait HasInner {
    type Inner;
}

impl<const M: usize> HasInner for [f32; M] {
    type Inner = Self;
}

impl<T: HasInner, const M: usize> HasInner for [T; M] {
    type Inner = T::Inner;
}
