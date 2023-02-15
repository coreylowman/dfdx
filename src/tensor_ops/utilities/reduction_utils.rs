#[cfg(feature = "cuda")]
use crate::prelude::{Axes, Shape};
#[cfg(feature = "cuda")]
use std::vec::Vec;

/// Moves all axes in Ax to the end of dims and strides and removes broadcasted dimensions
/// so that a cuda kernel called for each physical element of the input tensor will place elements
/// to be reduced with each other next to each other in memory.
#[cfg(feature = "cuda")]
pub(crate) fn permute_for_reductions<I, Ax: Axes>(dims: I, strides: I) -> (Vec<usize>, Vec<usize>)
where
    I: IntoIterator<Item = usize>,
{
    let mut tmp = dims
        .into_iter()
        .zip(strides.into_iter())
        .map(|x| (false, x))
        .collect::<Vec<_>>();

    for i in Ax::as_array().into_iter() {
        tmp[i as usize].0 = true;
    }

    // requires stable sorting to keep non-summed axes in the correct order
    tmp.sort_by_key(|x| x.0);

    tmp.into_iter()
        .map(|(_, x)| x)
        .filter(|(_, stride)| *stride != 0)
        .unzip()
}

/// Returns the physical number of elements and strides of dst so that broadcasted dimensions in
/// src are also broadcasted in dst
#[cfg(feature = "cuda")]
pub(crate) fn reduction_output_strides<Ax: Axes, Src: Shape, Dst: Shape>(
    src_strides: Src::Concrete,
    dst: Dst,
) -> (usize, Dst::Concrete) {
    let dst_dims = dst.concrete();
    let mut dst_strides = dst.strides();
    let mut numel = 1;
    let mut j = Dst::NUM_DIMS;

    for i in (0..Src::NUM_DIMS).rev() {
        if !Ax::as_array().into_iter().any(|x| x as usize == i) {
            j -= 1;
            if src_strides[i] == 0 {
                dst_strides[j] = 0;
            } else {
                dst_strides[j] = numel;
                numel *= dst_dims[j];
            }
        }
    }

    (numel, dst_strides)
}

/// Gives the product of all dimensions that are being reduced and are broadcasted.
#[cfg(feature = "cuda")]
pub(crate) fn reduction_elems_per_thread<Ax: Axes, S: Shape>(
    dims: S::Concrete,
    strides: S::Concrete,
) -> usize {
    Ax::as_array()
        .into_iter()
        .map(|ax| {
            if strides[ax as usize] == 0 {
                dims[ax as usize]
            } else {
                1
            }
        })
        .product()
}
