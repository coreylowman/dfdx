use crate::{shapes::*, tensor::*};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

/// Concatenate two tensors along the first dimension.
///
/// **Pytorch equivalent** `torch.concat`.
///
/// Concat with const dims **requires nightly**:
/// ```ignore
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
/// let b: Tensor<Rank2<3, 4>, f32, _> = dev.zeros();
/// let _: Tensor<Rank2<6, 4>, f32, _> = a.concat(b);
/// ```
///
/// Concat with usize dims:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a: Tensor<(usize, Const<3>), f32, _> = dev.zeros_like(&(2, Const));
/// let b: Tensor<(usize, Const<3>), f32, _> = dev.zeros_like(&(4, Const));
/// let c: Tensor<(usize, Const<3>), f32, _> = a.concat(b);
/// assert_eq!(c.shape().0, 6);
/// ```
#[deprecated = "Use TryConcatAlong instead"]
pub trait TryConcat<Rhs>: HasErr {
    type Output;

    /// Concatenate two tensors along the first dimension.
    #[deprecated = "Use TryConcatAlong::concat_along instead"]
    #[allow(deprecated)]
    fn concat(self, rhs: Rhs) -> Self::Output {
        #[allow(deprecated)]
        self.try_concat(rhs).unwrap()
    }

    /// Fallible version of [TryConcat::concat].
    #[deprecated = "Use TryConcatAlong::try_concat_along instead"]
    #[allow(deprecated)]
    fn try_concat(self, rhs: Rhs) -> Result<Self::Output, Self::Err>;
}

#[allow(deprecated)]
impl<A: Shape, B: Shape, T, R, E: Dtype, D: ConcatKernel<E>> TryConcat<Tensor<B, E, D, R>>
    for Tensor<A, E, D, T>
where
    A: ConcatShape<B>,
    T: Tape<E, D> + Merge<R>,
    R: Tape<E, D>,
{
    type Output = Tensor<A::Catted, E, D, T>;
    #[allow(deprecated)]
    fn try_concat(self, rhs: Tensor<B, E, D, R>) -> Result<Self::Output, Self::Err> {
        assert_eq!(
            self.strides,
            self.shape.strides(),
            "Concat requires contiguous tensors"
        );
        assert_eq!(
            rhs.strides,
            rhs.shape.strides(),
            "Concat requires contiguous tensors"
        );
        let (lhs, a_tape) = self.split_tape();
        let (rhs, b_tape) = rhs.split_tape();
        let mut tape = a_tape.merge(b_tape);
        let device = lhs.device.clone();
        let out = device.forward(&lhs, &rhs)?;
        let lhs_ghost = lhs.ghost();
        let rhs_ghost = rhs.ghost();
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_a, grad_b, grad_out) = grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            device.backward(grad_a, grad_b, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

pub trait ConcatKernel<E: Dtype>: Storage<E> {
    fn forward<A: Shape, B: Shape>(
        &self,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
    ) -> Result<Tensor<A::Catted, E, Self>, Self::Err>
    where
        A: ConcatShape<B>;
    fn backward(
        &self,
        grad_a: &mut Self::Vec,
        grad_b: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err>;
}

pub trait ConcatShape<Rhs: Shape>: Shape {
    type Catted: Shape;
    fn concat_shape(&self, rhs: &Rhs) -> Self::Catted;
}

macro_rules! impl_concat {
    ([$($Dims:tt $Idx:tt),*]) => {
        impl<A: Dim, B: Dim, $($Dims: Dim, )*> ConcatShape<(A, $($Dims, )*)>
            for (B, $($Dims, )*)
        where
            A: std::ops::Add<B>,
            <A as std::ops::Add<B>>::Output: Dim,
        {
            type Catted = (<A as std::ops::Add<B>>::Output, $($Dims, )*);

            fn concat_shape(&self, rhs: &(A, $($Dims, )*)) -> Self::Catted {
                $(assert_eq!(self.$Idx, rhs.$Idx);)*
                (rhs.0 + self.0, $(self.$Idx, )*)
            }
        }
    };
}

impl_concat!([]);
impl_concat!([D1 1]);
impl_concat!([D1 1, D2 2]);
impl_concat!([D1 1, D2 2, D3 3]);
impl_concat!([D1 1, D2 2, D3 3, D4 4]);
impl_concat!([D1 1, D2 2, D3 3, D4 4, D5 5]);

impl<const N: usize> ConcatShape<[usize; N]> for [usize; N]
where
    [usize; N]: Shape,
{
    type Catted = [usize; N];

    fn concat_shape(&self, rhs: &[usize; N]) -> [usize; N] {
        assert_eq!(self[1..], rhs[1..]);
        let mut out = *self;
        out[0] = self[0] + rhs[0];
        out
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_concat() {
        let dev: TestDevice = Default::default();
        let a: Tensor<(usize, Const<5>, Const<3>), TestDtype, _> =
            dev.sample_normal_like(&(3, Const, Const));
        let b: Tensor<(usize, Const<5>, Const<3>), TestDtype, _> =
            dev.sample_normal_like(&(2, Const, Const));
        let c = a.leaky_trace().concat(b.clone());
        assert_eq!(c.shape, (5, Const::<5>, Const::<3>));
        let c_vec = c.as_vec();
        assert_eq!(c_vec[..a.shape.num_elements()], a.as_vec());
        assert_eq!(c_vec[a.shape.num_elements()..], b.as_vec());
        let concat_grads = c.exp().sum().backward();
        let a_grads = a.leaky_trace().exp().sum().backward();
        let b_grads = b.leaky_trace().exp().sum().backward();
        assert_eq!(concat_grads.get(&a).as_vec(), a_grads.get(&a).as_vec());
        assert_eq!(concat_grads.get(&b).as_vec(), b_grads.get(&b).as_vec());
    }

    #[test]
    fn test_concat_shape() {
        let a: (usize, Const<5>) = (5, Const);
        let b: (usize, Const<5>) = (3, Const);
        assert_eq!(a.concat_shape(&b), (8, Const::<5>));

        let a: (Const<5>, Const<5>) = (Const, Const);
        let b: (usize, Const<5>) = (3, Const);
        assert_eq!(a.concat_shape(&b), (8, Const::<5>));

        let a: (usize, Const<5>) = (5, Const);
        let b: (Const<3>, Const<5>) = (Const, Const);
        assert_eq!(a.concat_shape(&b), (8, Const::<5>));

        #[cfg(feature = "nightly")]
        {
            let a: (Const<5>, Const<5>) = (Const, Const);
            let b: (Const<3>, Const<5>) = (Const, Const);
            assert_eq!(a.concat_shape(&b), (Const::<8>, Const::<5>));
        }
    }

    #[test]
    #[should_panic = "left: `10`,\n right: `7`"]
    fn test_concat_shape_fails() {
        (5, 10).concat_shape(&(3, 7));
    }
}
