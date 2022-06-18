use crate::prelude::*;
use std::ops::Neg;

/// Add together two [Tensor]s by broadcasting `rhs` `M` times, where `M` is the first dimension of `lhs`.
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
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoTape>,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
{
    fn f(x: &f32, y: &f32) -> f32 {
        x + y
    }
    fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    fn dfdy(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    binary_map_broadcast_rhs_first(lhs, rhs, f, dfdx, dfdy)
}

/// Subtract two [Tensor]s by broadcasting `rhs` `M` times, where `M` is the first dimension of `lhs`.
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
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoTape>,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
{
    fn f(x: &f32, y: &f32) -> f32 {
        x - y
    }
    fn dfdx(_x: &f32, _y: &f32) -> f32 {
        1.0
    }
    fn dfdy(_x: &f32, _y: &f32) -> f32 {
        -1.0
    }
    binary_map_broadcast_rhs_first(lhs, rhs, f, dfdx, dfdy)
}

/// Multiplies two [Tensor]s by broadcasting `rhs` `M` times, where `M` is the first dimension of `lhs`.
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
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoTape>,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
{
    fn f(x: &f32, y: &f32) -> f32 {
        x * y
    }
    fn dfdx(_x: &f32, y: &f32) -> f32 {
        *y
    }
    fn dfdy(x: &f32, _y: &f32) -> f32 {
        *x
    }
    binary_map_broadcast_rhs_first(lhs, rhs, f, dfdx, dfdy)
}

/// Divides two [Tensor]s by broadcasting `rhs` `M` times, where `M` is the first dimension of `lhs`.
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
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoTape>,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
{
    fn f(x: &f32, y: &f32) -> f32 {
        x * y.recip()
    }
    fn dfdx(_x: &f32, y: &f32) -> f32 {
        y.recip()
    }
    fn dfdy(x: &f32, y: &f32) -> f32 {
        x.neg() * y.powi(2).recip()
    }
    binary_map_broadcast_rhs_first(lhs, rhs, f, dfdx, dfdy)
}

/// Apply binary function `f` to `lhs` and `rhs`, where `rhs` is broadcasted `M` times to be the same shape as `lhs`.
/// `dfdx` and `dfdy` are the partial derivatives of f wrt. x and y respectively.
///
/// `f`, `dfdx`, and `dfdy` are all the same type.
///
/// Generics:
/// - `M`: The first dimension of `lhs`.
fn binary_map_broadcast_rhs_first<const M: usize, Lhs, Rhs, F, Dfdx, Dfdy>(
    lhs: Lhs,
    rhs: &Rhs,
    mut f: F,
    mut dfdx: Dfdx,
    mut dfdy: Dfdy,
) -> Lhs
where
    Rhs: 'static + Tensor<Dtype = f32, Tape = NoTape>,
    Lhs: Tensor<Dtype = f32, Array = [Rhs::Array; M]>,
    F: FnMut(&f32, &f32) -> f32,
    Dfdx: FnMut(&f32, &f32) -> f32,
    Dfdy: FnMut(&f32, &f32) -> f32,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
{
    let result = Lhs::NoTape::new_boxed(Lhs::Device::broadcast_rhs_first(
        lhs.data(),
        rhs.data(),
        &mut f,
    ));

    let (mut lhs, mut tape) = lhs.split_tape();
    let _rhs = rhs.phantom();
    let _result = result.phantom();

    // calculate derivatives
    let mut rhs_deriv: Box<Lhs::Array> =
        Lhs::Device::broadcast_rhs_first(lhs.data(), rhs.data(), &mut dfdy);
    Lhs::Device::broadcast_rhs_first_assign(lhs.mut_data(), rhs.data(), &mut |l, r| {
        *l = dfdx(l, r)
    });

    tape.add_backward_op(move |grads| {
        let result_grad: &Lhs::Array = grads.ref_gradient(&_result);
        // chain rule
        Lhs::Device::mul_assign(lhs.mut_data(), result_grad);
        Lhs::Device::mul_assign(rhs_deriv.as_mut(), result_grad);

        // sum first dimension
        let mut d_grad_rhs: Box<Rhs::Array> = Lhs::Device::zeros();
        for i in 0..M {
            Rhs::Device::add_assign(d_grad_rhs.as_mut(), &rhs_deriv[i]);
        }

        // gather gradients
        Lhs::Device::add_assign(grads.mut_gradient(&lhs), lhs.data());
        Rhs::Device::add_assign(grads.mut_gradient(&_rhs), d_grad_rhs.as_ref());
    });
    result.put_tape(tape)
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
                [1.23150945, 1.23150945, 1.23150945],
                [0.16666667, 9.09969139, 0.0030526067]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[-1.39817607, -10.33120060, -1.23456204]
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
                [
                    [-1.18180001, 0.2183, 0.2464],
                    [-0.34290004, 1.06799996, 1.00699997]
                ],
                [
                    [1.02110004, 0.12320001, 0.73430002],
                    [-0.022600055, 2.60579991, 0.85210001]
                ],
                [
                    [-0.49520001, -0.17839998, 0.28979999],
                    [-1.99530005, 1.27909994, 0.2313]
                ],
                [
                    [1.52859998, 0.2167, -0.85640001],
                    [-2.08930016, 2.41429996, -0.18920001]
                ]
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
            &[
                [-0.06131324, 0., 3.34758949],
                [-0.45304698, 0., 0.008297845]
            ]
        );
        assert_eq!(gradients.ref_gradient(&b), &[-0.39173374, 1., 10.01787472]);
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
                    [0.72567898, 0.71964973, -0.10490200]
                ],
                [
                    [0.022522321, 0.01397433, -0.11029122],
                    [1.05891907, -1.64302599, -0.086623795]
                ],
                [
                    [-0.01022976, 0.07275617, 0.12249343],
                    [-0.99347800, 0.39531571, -0.01336940]
                ],
                [
                    [0.03348432, -0.0042488202, 0.72275835],
                    [-1.09127557, -1.34880555, 0.03624960]
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
                [0.03996250, -0.016658332, -0.07002917],
                [-0.012020834, 0.05090000, 0.059550002]
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
                [-0.06131324, f32::INFINITY, 3.34758949],
                [-0.45304698, f32::INFINITY, 0.008297845]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[0.39173374, f32::NEG_INFINITY, -10.01787472]
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
                [-85.65350342, 0.43853861, 0.25533706],
                [0.011105392, -0.021563001, -4.27678776]
            ]
        );
    }
}
