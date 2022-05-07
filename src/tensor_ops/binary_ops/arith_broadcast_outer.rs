use crate::prelude::*;

// TODO abstract these all together somehow

pub fn broadcast_outer_add<Lhs, Rhs>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Rhs::ArrayType: Array,
    Lhs::ArrayType: Array<Element = Rhs::ArrayType>,
{
    let mut result = Lhs::ArrayType::ZEROS;
    for i in 0..Lhs::ArrayType::SIZE {
        result[i] = lhs.data()[i].add(rhs.data());
    }

    let lhs_deriv = lhs.data().map_elems(|_| 1.0);
    let rhs_deriv = rhs.data().map_elems(|_| 1.0);

    let result = Lhs::NoTape::new(result);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad_lhs = lhs_deriv.mul(tape.ref_gradient(&_result));
        tape.mut_gradient(&lhs).add_assign(&d_grad_lhs);

        let mut d_grad_rhs = Rhs::ArrayType::ZEROS;
        for i in 0..Lhs::ArrayType::SIZE {
            d_grad_rhs.add_assign(&rhs_deriv.mul(&tape.ref_gradient(&_result)[i]));
        }
        tape.mut_gradient(&_rhs).add_assign(&d_grad_rhs);
    });
    result.with_tape_holder(tape_holder)
}

pub fn broadcast_outer_sub<Lhs, Rhs>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Rhs::ArrayType: Array,
    Lhs::ArrayType: Array<Element = Rhs::ArrayType>,
{
    let mut result = Lhs::ArrayType::ZEROS;
    for i in 0..Lhs::ArrayType::SIZE {
        result[i] = lhs.data()[i].sub(rhs.data());
    }

    let lhs_deriv = lhs.data().map_elems(|_| 1.0);
    let rhs_deriv = rhs.data().map_elems(|_| -1.0);

    let result = Lhs::NoTape::new(result);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let d_grad_lhs = lhs_deriv.mul(tape.ref_gradient(&_result));
        tape.mut_gradient(&lhs).add_assign(&d_grad_lhs);

        let mut d_grad_rhs = Rhs::ArrayType::ZEROS;
        for i in 0..Lhs::ArrayType::SIZE {
            d_grad_rhs.add_assign(&rhs_deriv.mul(&tape.ref_gradient(&_result)[i]));
        }
        tape.mut_gradient(&_rhs).add_assign(&d_grad_rhs);
    });
    result.with_tape_holder(tape_holder)
}

pub fn broadcast_outer_mul<Lhs, Rhs>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Rhs::ArrayType: Array,
    Lhs::ArrayType: Array<Element = Rhs::ArrayType>,
{
    let mut result = Lhs::ArrayType::ZEROS;
    for i in 0..Lhs::ArrayType::SIZE {
        result[i] = lhs.data()[i].mul(rhs.data());
    }
    let lhs_deriv = rhs.data().clone();
    let rhs_deriv = lhs.data().clone();

    let result = Lhs::NoTape::new(result);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let mut d_grad_lhs = Lhs::ArrayType::ZEROS;
        for i in 0..Lhs::ArrayType::SIZE {
            d_grad_lhs[i] = lhs_deriv.mul(&tape.ref_gradient(&_result)[i]);
        }
        tape.mut_gradient(&lhs).add_assign(&d_grad_lhs);

        let mut d_grad_rhs = Rhs::ArrayType::ZEROS;
        let total = rhs_deriv.mul(&tape.ref_gradient(&_result));
        for i in 0..Lhs::ArrayType::SIZE {
            d_grad_rhs.add_assign(&total[i]);
        }
        tape.mut_gradient(&_rhs).add_assign(&d_grad_rhs);
    });
    result.with_tape_holder(tape_holder)
}

pub fn broadcast_outer_div<Lhs, Rhs>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Rhs::ArrayType: Array,
    Lhs::ArrayType: Array<Element = Rhs::ArrayType>,
{
    let mut result = Lhs::ArrayType::ZEROS;
    for i in 0..Lhs::ArrayType::SIZE {
        result[i] = lhs.data()[i].div(rhs.data());
    }
    let lhs_deriv = rhs.data().mapv_elems(f32::recip);
    let mut rhs_deriv = Lhs::ArrayType::ZEROS;
    for i in 0..Lhs::ArrayType::SIZE {
        rhs_deriv[i] = lhs.data()[i].zip_map(rhs.data(), BinaryDiv::dfdy);
    }

    let result = Lhs::NoTape::new(result);
    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let mut d_grad_lhs = Lhs::ArrayType::ZEROS;
        for i in 0..Lhs::ArrayType::SIZE {
            d_grad_lhs[i] = lhs_deriv.mul(&tape.ref_gradient(&_result)[i]);
        }
        tape.mut_gradient(&lhs).add_assign(&d_grad_lhs);

        let mut d_grad_rhs = Rhs::ArrayType::ZEROS;
        let total = rhs_deriv.mul(&tape.ref_gradient(&_result));
        for i in 0..Lhs::ArrayType::SIZE {
            d_grad_rhs.add_assign(&total[i]);
        }
        tape.mut_gradient(&_rhs).add_assign(&d_grad_rhs);
    });
    result.with_tape_holder(tape_holder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_outer_add() {
        let a: Tensor2D<3, 5> = Tensor2D::ones();
        let b: Tensor1D<5> = Tensor1D::ones();
        let r = broadcast_outer_add(a.trace(), &b);
        assert_eq!(r.data(), &[[2.0; 5]; 3]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[1.0 / 15.0; 5]; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[0.20000002; 5]);
    }

    #[test]
    fn test_broadcast_outer_sub() {
        let a: Tensor2D<3, 5> = Tensor2D::ones();
        let b: Tensor1D<5> = Tensor1D::ones();
        let r = broadcast_outer_sub(a.trace(), &b);
        assert_eq!(r.data(), &[[0.0; 5]; 3]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[[1.0 / 15.0; 5]; 3]);
        assert_eq!(gradients.ref_gradient(&b), &[-0.20000002; 5]);
    }

    #[test]
    fn test_broadcast_outer_mul() {
        let a = Tensor2D::new([
            [0.3230, 0.8566, 0.3156, 0.5860, 0.8327],
            [0.4928, 0.1055, 0.4153, 0.0283, 0.0722],
            [0.6562, 0.5700, 0.0569, 0.3314, 0.2639],
        ]);
        let b = Tensor1D::new([0.6294, 0.4542, 0.1578, 0.1773, 0.6875]);
        let r = broadcast_outer_mul(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [0.20329621, 0.3890677, 0.04980168, 0.10389781, 0.5724813],
                [0.31016833, 0.0479181, 0.065534346, 0.0050175902, 0.0496375],
                [0.4130123, 0.258894, 0.00897882, 0.058757223, 0.18143126]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[[
                0.041960005,
                0.030280001,
                0.010520001,
                0.011820001,
                0.045833334
            ]; 3]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[0.09813334, 0.10214001, 0.052520003, 0.06304667, 0.077920005]
        );
    }

    #[test]
    fn test_broadcast_outer_div() {
        let a = Tensor2D::new([
            [0.3230, 0.8566, 0.3156, 0.5860, 0.8327],
            [0.4928, 0.1055, 0.4153, 0.0283, 0.0722],
            [0.6562, 0.5700, 0.0569, 0.3314, 0.2639],
        ]);
        let b = Tensor1D::new([0.6294, 0.4542, 0.1578, 0.1773, 0.6875]);
        let r = broadcast_outer_div(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [0.51318717, 1.8859533, 2.0, 3.3051326, 1.2112],
                [0.78296787, 0.23227653, 2.6318123, 0.15961647, 0.10501818],
                [1.0425802, 1.2549537, 0.360583, 1.8691483, 0.38385457]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[[0.105920985, 0.14677823, 0.4224757, 0.37601054, 0.0969697]; 3]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[-0.24772115, -0.49510992, -2.109166, -2.0056014, -0.16485555]
        );
    }
}
