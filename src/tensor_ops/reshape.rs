use crate::prelude::*;

pub fn reshape<T, R>(t: T) -> R
where
    T: Tensor<Dtype = f32>,
    R: Tensor<Tape = T::Tape, Dtype = f32>,
    ConstEq<{ T::NUM_ELEMENTS }, { R::NUM_ELEMENTS }>: ConstAssert<Result = True>,
{
    let (mut t, mut tape) = t.split_tape();
    let mut result: R::NoTape = R::NoTape::zeros();
    copy(t.data(), result.mut_data());
    let _result = result.phantom();
    tape.add_backward_op(move |grads| {
        let result_grad: &R::Array = grads.ref_gradient(&_result);
        copy(result_grad, t.mut_data());
        T::Device::add(grads.mut_gradient(&t), t.data());
    });
    result.put_tape(tape)
}

fn copy<Lhs: CountElements, Rhs: CountElements<Dtype = Lhs::Dtype>>(lhs: &Lhs, rhs: &mut Rhs)
where
    ConstEq<{ Lhs::NUM_ELEMENTS }, { Rhs::NUM_ELEMENTS }>: ConstAssert<Result = True>,
{
    let l = lhs.ref_first_elem() as *const Lhs::Dtype;
    let r = rhs.mut_first_elem() as *mut Lhs::Dtype;
    unsafe {
        std::ptr::copy_nonoverlapping(l, r, Lhs::NUM_ELEMENTS);
    }
}

struct True;
trait ConstAssert {
    type Result;
}

struct ConstEq<const A: usize, const B: usize>;
impl<const N: usize> ConstAssert for ConstEq<N, N> {
    type Result = True;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_reshape() {
        let a = Tensor0D::new(3.14);
        let b: Tensor1D<1> = reshape(a.duplicate());
        assert_eq!(b.data(), &[3.14]);

        let c: Tensor2D<1, 1> = reshape(a.duplicate());
        assert_eq!(c.data(), &[[3.14]]);
    }

    #[test]
    fn test_1d_reshape() {
        let a = Tensor1D::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let b: Tensor2D<2, 3, OwnedTape> = reshape(a.trace());
        assert_eq!(b.data(), &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let gradients = b.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[0.18419516, 0.20356713, 0.22497648, 0.24863747, 0.2747869, 0.3036865]
        )
    }
}
