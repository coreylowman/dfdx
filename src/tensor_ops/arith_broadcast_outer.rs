use super::binary_map::{
    add, binary_map_broadcast_rhs_first, binary_map_broadcast_rhs_first_2d, div, mul, sub,
};
use crate::prelude::*;

/// `lhs + &rhs`. `rhs` is broadcasted `M` times, where `M` is the first dimension of `lhs`.
///
/// E.g If Lhs has dimension `(2, 3)`, then Rhs has to be dimension `(3,)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let b = Tensor1D::new([-1.0, 0.0, 1.0]);
/// let r = add_broadcast_rhs_first(a, &b);
/// assert_eq!(r.data(), &[[0.0, 2.0, 4.0], [3.0, 5.0, 7.0]]);
/// ```
pub fn add_broadcast_rhs_first<Lhs, Rhs, const M: usize>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor<Array = [Rhs::Array; M], Dtype = f32>,
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoneTape>,
{
    binary_map_broadcast_rhs_first(lhs, rhs, add::f, add::dfdx, add::dfdy)
}

/// `lhs - &rhs`. `rhs` is broadcasted `M` times, where `M` is the first dimension of `lhs`.
///
/// E.g If Lhs has dimension `(2, 3)`, then Rhs has to be dimension `(3,)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let b = Tensor1D::new([-1.0, 0.0, 1.0]);
/// let r = sub_broadcast_rhs_first(a, &b);
/// assert_eq!(r.data(), &[[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]]);
/// ```
pub fn sub_broadcast_rhs_first<Lhs, Rhs, const M: usize>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor<Array = [Rhs::Array; M], Dtype = f32>,
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoneTape>,
{
    binary_map_broadcast_rhs_first(lhs, rhs, sub::f, sub::dfdx, sub::dfdy)
}

/// `lhs * &rhs`. `rhs` is broadcasted `M` times, where `M` is the first dimension of `lhs`.
///
/// E.g If Lhs has dimension `(2, 3)`, then Rhs has to be dimension `(3,)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let b = Tensor1D::new([-1.0, 0.0, 1.0]);
/// let r = mul_broadcast_rhs_first(a, &b);
/// assert_eq!(r.data(), &[[-1.0, 0.0, 3.0], [-4.0, 0.0, 6.0]]);
/// ```
pub fn mul_broadcast_rhs_first<Lhs, Rhs, const M: usize>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor<Array = [Rhs::Array; M], Dtype = f32>,
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoneTape>,
{
    binary_map_broadcast_rhs_first(lhs, rhs, mul::f, mul::dfdx, mul::dfdy)
}

/// `lhs / &rhs`. `rhs` is broadcasted `M` times, where `M` is the first dimension of `lhs`.
///
/// E.g If Lhs has dimension `(2, 3)`, then Rhs has to be dimension `(3,)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor2D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// let b = Tensor1D::new([-1.0, 0.0, 1.0]);
/// let r = div_broadcast_rhs_first(a, &b);
/// assert_eq!(r.data(), &[[-1.0, f32::INFINITY, 3.0], [-4.0, f32::INFINITY, 6.0]]);
/// ```
pub fn div_broadcast_rhs_first<Lhs, Rhs, const M: usize>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor<Array = [Rhs::Array; M], Dtype = f32>,
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoneTape>,
{
    binary_map_broadcast_rhs_first(lhs, rhs, div::f, div::dfdx, div::dfdy)
}

/// `lhs + &rhs`. `rhs` is broadcasted `N * M` times, where `N` is the first dimension and `N` is the second dimension of `lhs`.
///
/// E.g If Lhs has dimension `(5, 2, 3)`, then Rhs has to be dimension `(3,)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor3D::new([
///     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
///     [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
/// ]);
/// let b = Tensor1D::new([-1.0, 0.0, 1.0]);
/// let r = add_broadcast_rhs_first_2d(a, &b);
/// assert_eq!(r.data(), &[
///     [[0.0, 2.0, 4.0], [3.0, 5.0, 7.0]],
///     [[-2.0, -2.0, -2.0], [-5.0, -5.0, -5.0]]
/// ]);
/// ```
pub fn add_broadcast_rhs_first_2d<Lhs, Rhs, const M: usize, const N: usize>(
    lhs: Lhs,
    rhs: &Rhs,
) -> Lhs
where
    Lhs: Tensor<Array = [[Rhs::Array; M]; N], Dtype = f32>,
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoneTape>,
{
    binary_map_broadcast_rhs_first_2d(lhs, rhs, add::f, add::dfdx, add::dfdy)
}

/// `lhs - &rhs`. `rhs` is broadcasted `N * M` times, where `N` is the first dimension and `N` is the second dimension of `lhs`.
///
/// E.g If Lhs has dimension `(5, 2, 3)`, then Rhs has to be dimension `(3,)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor3D::new([
///     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
///     [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
/// ]);
/// let b = Tensor1D::new([-1.0, 0.0, 1.0]);
/// let r = sub_broadcast_rhs_first_2d(a, &b);
/// assert_eq!(r.data(), &[
///     [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
///     [[-0.0, -2.0, -4.0], [-3.0, -5.0, -7.0]]
/// ]);
/// ```
pub fn sub_broadcast_rhs_first_2d<Lhs, Rhs, const M: usize, const N: usize>(
    lhs: Lhs,
    rhs: &Rhs,
) -> Lhs
where
    Lhs: Tensor<Array = [[Rhs::Array; M]; N], Dtype = f32>,
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoneTape>,
{
    binary_map_broadcast_rhs_first_2d(lhs, rhs, sub::f, sub::dfdx, sub::dfdy)
}

/// `lhs * &rhs`. `rhs` is broadcasted `N * M` times, where `N` is the first dimension and `N` is the second dimension of `lhs`.
///
/// E.g If Lhs has dimension `(5, 2, 3)`, then Rhs has to be dimension `(3,)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor3D::new([
///     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
///     [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
/// ]);
/// let b = Tensor1D::new([-1.0, 0.0, 1.0]);
/// let r = mul_broadcast_rhs_first_2d(a, &b);
/// assert_eq!(r.data(), &[
///     [[-1.0, 0.0, 3.0], [-4.0, 0.0, 6.0]],
///     [[1.0, 0.0, -3.0], [4.0, 0.0, -6.0]]
/// ]);
/// ```
pub fn mul_broadcast_rhs_first_2d<Lhs, Rhs, const M: usize, const N: usize>(
    lhs: Lhs,
    rhs: &Rhs,
) -> Lhs
where
    Lhs: Tensor<Array = [[Rhs::Array; M]; N], Dtype = f32>,
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoneTape>,
{
    binary_map_broadcast_rhs_first_2d(lhs, rhs, mul::f, mul::dfdx, mul::dfdy)
}

/// `lhs / &rhs`. `rhs` is broadcasted `N * M` times, where `N` is the first dimension and `N` is the second dimension of `lhs`.
///
/// E.g If Lhs has dimension `(5, 2, 3)`, then Rhs has to be dimension `(3,)`.
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// let a = Tensor3D::new([
///     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
///     [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
/// ]);
/// let b = Tensor1D::new([-1.0, 0.0, 1.0]);
/// let r = div_broadcast_rhs_first_2d(a, &b);
/// assert_eq!(r.data(), &[
///     [[-1.0, f32::INFINITY, 3.0], [-4.0, f32::INFINITY, 6.0]],
///     [[1.0, -f32::INFINITY, -3.0], [4.0, -f32::INFINITY, -6.0]]
/// ]);
/// ```
pub fn div_broadcast_rhs_first_2d<Lhs, Rhs, const M: usize, const N: usize>(
    lhs: Lhs,
    rhs: &Rhs,
) -> Lhs
where
    Lhs: Tensor<Array = [[Rhs::Array; M]; N], Dtype = f32>,
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoneTape>,
{
    binary_map_broadcast_rhs_first_2d(lhs, rhs, div::f, div::dfdx, div::dfdy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_broadcast_rhs_first_1d() {
        let a = Tensor1D::new([-1.0, 0.0, 1.0]);
        let b = Tensor0D::new(1.0);
        let r = add_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(r.data(), &[0.0, 1.0, 2.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &1.0);
    }

    #[test]
    fn test_add_broadcast_rhs_first_2d() {
        let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, 4.0, -3.0]]);
        let b = Tensor1D::new([-1.0, 0.0, 1.0]);
        let r = add_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(r.data(), &[[0.0, 2.0, 4.0], [-2.0, 4.0, -2.0]]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.16666667, 1.2315094, 9.099691],
                [0.02255588, 9.099691, 0.02255588]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[0.18922254, 10.331201, 9.122248]
        );
    }

    #[test]
    fn test_add_broadcast_rhs_first_3d() {
        let a = Tensor3D::new([
            [[-1.1602, 0.0234, -0.2773], [0.6975, -0.4684, 0.8890]],
            [[1.0427, -0.0717, 0.2106], [1.0178, 1.0694, 0.7341]],
            [[-0.4736, -0.3733, -0.2339], [-0.9549, -0.2573, 0.1133]],
            [[1.5502, 0.0218, -1.3801], [-1.0489, 0.8779, -0.3072]],
        ]);
        let b = Tensor2D::new([[0.0216, -0.1949, -0.5237], [1.0404, -1.5364, -0.1180]]);
        let r = add_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [[-1.1386, -0.17150001, -0.801], [1.7379, -2.0047998, 0.771]],
                [
                    [1.0643001, -0.2666, -0.31309998],
                    [2.0582, -0.467, 0.61609995]
                ],
                [[-0.452, -0.5682, -0.7576], [0.0855, -1.7937, -0.0046999976]],
                [
                    [1.5718, -0.17310001, -1.9038],
                    [-0.00849998, -0.65849996, -0.42520002]
                ],
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[[1.0 / 24.0; 3]; 2]; 4]);
        assert_eq!(gradients.ref_gradient(&b), &[[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_sub_broadcast_rhs_first_1d() {
        let a = Tensor1D::new([-1.0, 0.0, 1.0]);
        let b = Tensor0D::new(1.0);
        let r = sub_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(r.data(), &[-2.0, -1.0, 0.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &-1.0);
    }

    #[test]
    fn test_sub_broadcast_rhs_first_2d() {
        let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, 4.0, -3.0]]);
        let b = Tensor1D::new([-1.0, 0.0, 1.0]);
        let r = sub_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(r.data(), &[[2., 2., 2.], [0., 4., -4.]]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [1.2315095, 1.2315095, 1.2315095],
                [0.16666667, 9.099691, 0.0030526067]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[-1.3981761, -10.331201, -1.234562]
        );
    }

    #[test]
    fn test_sub_broadcast_rhs_first_3d() {
        let a = Tensor3D::new([
            [[-1.1602, 0.0234, -0.2773], [0.6975, -0.4684, 0.8890]],
            [[1.0427, -0.0717, 0.2106], [1.0178, 1.0694, 0.7341]],
            [[-0.4736, -0.3733, -0.2339], [-0.9549, -0.2573, 0.1133]],
            [[1.5502, 0.0218, -1.3801], [-1.0489, 0.8779, -0.3072]],
        ]);
        let b = Tensor2D::new([[0.0216, -0.1949, -0.5237], [1.0404, -1.5364, -0.1180]]);
        let r = sub_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [[-1.1818, 0.2183, 0.2464], [-0.34290004, 1.068, 1.007]],
                [[1.0211, 0.12320001, 0.7343], [-0.022600055, 2.6058, 0.8521]],
                [[-0.4952, -0.17839998, 0.2898], [-1.9953, 1.2791, 0.2313]],
                [[1.5286, 0.2167, -0.8564], [-2.0893002, 2.4143, -0.18920001]]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[[1.0 / 24.0; 3]; 2]; 4]);
        assert_eq!(gradients.ref_gradient(&b), &[[-1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_mul_broadcast_rhs_first_1d() {
        let a = Tensor1D::new([-1.0, 0.0, 1.0]);
        let b = Tensor0D::new(1.0);
        let r = mul_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(r.data(), &[-1.0, 0.0, 1.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &0.0);
    }

    #[test]
    fn test_mul_broadcast_rhs_first_2d() {
        let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, 4.0, -3.0]]);
        let b = Tensor1D::new([-1.0, 0.0, 1.0]);
        let r = mul_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(r.data(), &[[-1., 0., 3.], [1., 0., -3.]]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[[-0.06131324, 0., 3.3475895], [-0.45304698, 0., 0.008297845]]
        );
        assert_eq!(gradients.ref_gradient(&b), &[-0.39173374, 1., 10.017875]);
    }

    #[test]
    fn test_mul_broadcast_rhs_first_3d() {
        let a = Tensor3D::new([
            [[-1.1602, 0.0234, -0.2773], [0.6975, -0.4684, 0.8890]],
            [[1.0427, -0.0717, 0.2106], [1.0178, 1.0694, 0.7341]],
            [[-0.4736, -0.3733, -0.2339], [-0.9549, -0.2573, 0.1133]],
            [[1.5502, 0.0218, -1.3801], [-1.0489, 0.8779, -0.3072]],
        ]);
        let b = Tensor2D::new([[0.0216, -0.1949, -0.5237], [1.0404, -1.5364, -0.1180]]);
        let r = mul_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [
                    [-0.02506032, -0.00456066, 0.14522201],
                    [0.725679, 0.71964973, -0.104902]
                ],
                [
                    [0.022522321, 0.01397433, -0.11029122],
                    [1.0589191, -1.643026, -0.086623795]
                ],
                [
                    [-0.01022976, 0.07275617, 0.12249343],
                    [-0.993478, 0.3953157, -0.0133694]
                ],
                [
                    [0.03348432, -0.0042488202, 0.72275835],
                    [-1.0912756, -1.3488056, 0.0362496]
                ]
            ]
        );
        let gradients = r.mean().backward();

        assert_eq!(
            gradients.ref_gradient(&a),
            &[[
                [0.00090000004, -0.008120834, -0.021820834],
                [0.043350004, -0.06401667, -0.004916667]
            ]; 4]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [0.0399625, -0.016658332, -0.07002917],
                [-0.012020834, 0.0509, 0.059550002]
            ]
        );
    }

    #[test]
    fn test_div_broadcast_rhs_first_1d() {
        let a = Tensor1D::new([-1.0, 0.0, 1.0]);
        let b = Tensor0D::new(1.0);
        let r = div_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(r.data(), &[-1.0, 0.0, 1.0]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[1.0 / 3.0; 3]);
        assert_eq!(gradients.ref_gradient(&b), &0.0);
    }

    #[test]
    fn test_div_broadcast_rhs_first_2d() {
        let a = Tensor2D::new([[1.0, 2.0, 3.0], [-1.0, 4.0, -3.0]]);
        let b = Tensor1D::new([-1.0, 0.0, 1.0]);
        let r = div_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[[-1., f32::INFINITY, 3.], [1., f32::INFINITY, -3.]]
        );
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [-0.06131324, f32::INFINITY, 3.3475895],
                [-0.45304698, f32::INFINITY, 0.008297845]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[0.39173374, f32::NEG_INFINITY, -10.017875]
        );
    }

    #[test]
    fn test_div_broadcast_rhs_first_3d() {
        let a = Tensor3D::new([
            [[-1.1602, 0.0234, -0.2773], [0.6975, -0.4684, 0.8890]],
            [[1.0427, -0.0717, 0.2106], [1.0178, 1.0694, 0.7341]],
            [[-0.4736, -0.3733, -0.2339], [-0.9549, -0.2573, 0.1133]],
            [[1.5502, 0.0218, -1.3801], [-1.0489, 0.8779, -0.3072]],
        ]);
        let b = Tensor2D::new([[0.0216, -0.1949, -0.5237], [1.0404, -1.5364, -0.1180]]);
        let r = div_broadcast_rhs_first(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [
                    [-53.712963, -0.12006156, 0.5295016],
                    [0.6704152, 0.30486852, -7.533898]
                ],
                [
                    [48.273148, 0.36788094, -0.40213865],
                    [0.97827756, -0.69604266, -6.221186]
                ],
                [
                    [-21.925926, 1.915341, 0.44662976],
                    [-0.9178201, 0.1674694, -0.9601695]
                ],
                [
                    [71.76852, -0.11185223, 2.6352875],
                    [-1.0081699, -0.5714007, 2.60339]
                ]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[[
                [1.9290123, -0.21378484, -0.0795621],
                [0.0400487, -0.027119674, -0.35310733]
            ]; 4]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [-85.6535, 0.4385386, 0.25533706],
                [0.011105392, -0.021563001, -4.2767878]
            ]
        );
    }
}
