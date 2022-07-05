//! Provides some generic functions to load & save Nd arrays in the [.npy](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html)
//! format. See [load()] and [save()]

mod load;
mod save;

pub use load::*;
pub use save::*;

const MAGIC_NUMBER: &[u8] = b"\x93NUMPY";
const VERSION: &[u8] = &[1, 0];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Big,
    Little,
    Native,
}

fn to_shape_str(shape: Vec<usize>) -> String {
    shape
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join(", ")
        + if shape.len() == 1 { "," } else { "" }
}

/// Represents the NumpyDtype as a const str value.
///
/// Values should match up to the [numpy documentation](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
/// for dtypes.
///
/// For example an f32's dtype is "f4".
pub trait NumpyDtype {
    const DTYPE: &'static str;
}

impl NumpyDtype for f32 {
    const DTYPE: &'static str = "f4";
}

impl NumpyDtype for f64 {
    const DTYPE: &'static str = "f8";
}

impl<T: NumpyDtype, const M: usize> NumpyDtype for [T; M] {
    const DTYPE: &'static str = T::DTYPE;
}

/// A type that implements this returns a vec of usize
/// that can represent a tuple of ints in a .npy file.
///
/// By default this function returns an empty vec, because
/// a single number is represented by the empty tuple in
/// a .npy file.
pub trait NumpyShape {
    fn shape() -> Vec<usize> {
        Vec::new()
    }
}

impl NumpyShape for f32 {}
impl NumpyShape for f64 {}

impl<T: NumpyShape, const M: usize> NumpyShape for [T; M] {
    fn shape() -> Vec<usize> {
        let mut s = T::shape();
        s.insert(0, M);
        s
    }
}
