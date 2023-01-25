#[cfg(feature = "cuda")]
use crate::prelude::Axes;
#[cfg(feature = "cuda")]
use std::vec::Vec;

/// Moves all axes in Ax to the end of dims and strides and removes broadcasted dimensions
/// so that a cuda kernel called for each physical element of the input tensor will place elements
/// to be reduced with each other next to each other in memory.
#[cfg(feature = "cuda")]
pub(super) fn permute_for_reductions<I, Ax: Axes>(dims: I, strides: I) -> (Vec<usize>, Vec<usize>)
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
