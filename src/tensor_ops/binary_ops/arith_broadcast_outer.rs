use crate::prelude::*;

// TODO abstract these all together somehow

fn bouter_add<T: CountElements, const M: usize>(lhs: &[T; M], rhs: &T, out: &mut [T; M])
where
    Cpu: ZipMapElements<T, T>,
{
    for i in 0..M {
        Cpu::zip_map_into(&lhs[i], rhs, &mut out[i], |x, y| x + y);
    }
}

pub fn broadcast_outer_add<Lhs, Rhs, const M: usize>(lhs: Lhs, rhs: &Rhs) -> Lhs
where
    Lhs: Tensor<Array = [Rhs::Array; M]>,
    Rhs: 'static + Tensor<TapeHolder = NoTape>,
    Lhs::Device: Device<Lhs::Array> + Device<Rhs::Array>,
    Cpu: ZipMapElements<Rhs::Array, Rhs::Array>,
{
    let result = Lhs::NoTape::new_boxed({
        let mut out: Box<Lhs::Array> = Lhs::Device::zeros();
        bouter_add(lhs.data(), rhs.data(), out.as_mut());
        out
    });

    let mut lhs_deriv = Lhs::Device::map(lhs.data(), |_| 1.0);

    let (lhs, mut tape_holder) = lhs.split_tape_holder();
    let _rhs = rhs.phantom();
    let _result = result.phantom();
    tape_holder.add_operation(move |tape| {
        let result_grad = tape.ref_gradient(&_result);

        // TODO we can get rid of this mul_assign since lhs_deriv is all 1
        Lhs::Device::mul_assign(lhs_deriv.as_mut(), result_grad);

        let mut d_grad_rhs: Box<Rhs::Array> = Lhs::Device::zeros();
        for i in 0..M {
            Lhs::Device::add_assign(d_grad_rhs.as_mut(), &result_grad[i]);
        }

        Lhs::Device::add_assign(tape.mut_gradient(&lhs), lhs_deriv.as_ref());
        Lhs::Device::add_assign(tape.mut_gradient(&_rhs), d_grad_rhs.as_ref());
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
}
