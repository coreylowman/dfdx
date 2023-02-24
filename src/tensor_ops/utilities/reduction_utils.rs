use crate::shapes::{Axes, Shape};
use crate::tensor::cpu::NdIndex;
#[cfg(feature = "cuda")]
use std::vec::Vec;

/// Permutes strides so that all reduces axes are rightmost.
/// For example reducing (2, 3, 4) to (4,) would permute the shape to (4, 2, 3)
/// via the strides.
///
/// This makes all the reduced elements sequential when indexed using [NdIndex].
#[inline(always)]
pub(crate) fn index_for_reductions<S: Shape, Ax: Axes>(
    shape: S,
    strides: S::Concrete,
) -> NdIndex<S> {
    let dims = shape.concrete();
    let mut new_shape: S::Concrete = Default::default();
    let mut new_strides: S::Concrete = Default::default();
    let num_non_reduced_dims = S::NUM_DIMS - Ax::as_array().into_iter().count();

    let mut i_reduced = 0;
    let mut i_non_reduced = 0;
    for i_src in 0..S::NUM_DIMS {
        if Ax::as_array().into_iter().any(|x| x == i_src as isize) {
            // this axis is reduced
            new_shape[num_non_reduced_dims + i_reduced] = dims[i_src];
            new_strides[num_non_reduced_dims + i_reduced] = strides[i_src];
            i_reduced += 1;
        } else {
            // this axis is not-reduced
            new_shape[i_non_reduced] = dims[i_src];
            new_strides[i_non_reduced] = strides[i_src];
            i_non_reduced += 1;
        }
    }
    NdIndex {
        indices: Default::default(),
        shape: new_shape,
        strides: new_strides,
        next: Some(0),
        contiguous: (new_shape == dims && new_strides == shape.strides())
            .then(|| shape.num_elements()),
    }
}

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
