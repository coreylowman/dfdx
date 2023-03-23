use crate::prelude::{Shape, Unit};

/// Create a 2D triangular mask with the given shape by setting excluded values to `E::default()`,
///
/// # Parameters
/// - `shape`: The shape of the resulting mask
///     - For shapes > 2 dimensions, this will repeat the 2D triangle over each extra dimension
///     - For shapes < 2 dimensions, this will linearly offset the mask in that dimension by `offset`
/// - `upper`: Whether this should create an upper or lower triangle mask
///     - Upper triangles will contain values at and above the offset diagonal and `E::default()` elsewhere
///     - Lower triangles will contain values at and below the offset diagonal and `E::default()` elsewhere
/// - `offset`: The offset from the main diagonal
///     - Positive values shift the values in the +M/-N direction
///     - Negative values shift the values in the -M/+N direction
pub fn triangle_mask<S: Shape, E: Unit>(data: &mut Vec<E>, shape: &S, upper: bool, offset: isize) {
    // Get the shape of the last two axes.
    let [num_rows, num_cols] = [
        (S::NUM_DIMS > 1)
            .then(|| shape.concrete()[S::NUM_DIMS - 2])
            .unwrap_or(1),
        (S::NUM_DIMS > 0)
            .then(|| shape.concrete()[S::NUM_DIMS - 1])
            .unwrap_or(1),
    ];
    let mat_size = num_rows * num_cols;

    // Get the first 2D matrix in this data. This will be copied to each subsequent matrix.
    let (mut mat2d, mut rest) = data.as_mut_slice().split_at_mut(mat_size);
    if upper {
        for r in (-offset).max(0) as usize..num_rows {
            for c in 0..((r as isize + offset).max(0) as usize).min(num_cols) {
                mat2d[r * num_cols + c] = E::default();
            }
        }
    } else {
        for r in 0..num_rows {
            for c in (r as isize + offset + 1).max(0) as usize..num_cols {
                mat2d[r * num_cols + c] = E::default();
            }
        }
    }
    while !rest.is_empty() {
        rest[..mat_size].copy_from_slice(mat2d);
        (mat2d, rest) = rest.split_at_mut(mat_size);
    }
}
