use std::sync::Arc;
use super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    shapes::{Dtype, Shape},
    tensor::cpu::{Cpu, StridedArray},
};

pub trait UnaryDerivative<E> {
    fn f(&self, x: &E) -> E;
    fn df(&self, x: &E) -> E;
}

pub trait BinaryDerivative<E> {
    fn f(&self, x: &E, y: &E) -> E;
    fn dfdx(&self, x: &E, y: &E) -> E;
    fn dfdy(&self, x: &E, y: &E) -> E;
}

impl<E: Dtype, Op: UnaryDerivative<E>> UnaryKernel<Op, E> for Cpu {
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let mut out: Self::Storage<S, E> = inp.clone();
        for x in out.buf_iter_mut() {
            *x = op.f(x);
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &Self::Storage<S, E>,
        grad_inp: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_inp.data.len(), grad_out.data.len());
        debug_assert_eq!(inp.data.len(), grad_out.data.len());
        for (i, x) in grad_inp.buf_iter_mut().enumerate() {
            *x += op.df(&inp.data[i]) * grad_out.data[i];
        }
        Ok(())
    }
}

/// Yields the amount to increment an index into a strided array when transitioning across a given
/// dimension
fn get_strided_incrs<S: Shape>(shape: S, strides: S::Concrete) -> S::Concrete {
    let mut out: S::Concrete = Default::default();
    let dims = shape.concrete();
    let mut elem_size = 1;

    for i in (0..S::NUM_DIMS).rev() {
        if strides[i] == 0 {
            out[i] = -(elem_size as isize - 1) as usize;
        } else {
            elem_size *= dims[i];
            out[i] = 1;
        }
    }

    out
}

/// outputs the dimension that the index is transitioning across
#[inline]
fn get_incr_dim<S: Shape>(out_i: usize, strides: S::Concrete) -> usize {
    strides
        .into_iter()
        .position(|stride| {
            if stride == 0 {
                false
            } else {
                out_i % stride == 0
            }
        })
        .unwrap_or(0)
}

#[inline]
fn incr_arg_i<S: Shape>(i: &mut usize, incrs: S::Concrete, dim: usize) {
    if dim >= S::NUM_DIMS {
        return;
    }
    *i = (*i as isize + incrs[dim] as isize) as usize;
}

impl<E: Dtype, Op: BinaryDerivative<E>> BinaryKernel<Op, E> for Cpu {
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let mut out: Self::Storage<S, E> = StridedArray::try_new_merge(lhs, rhs, E::default())?;
        let lhs_incrs = get_strided_incrs(lhs.shape, lhs.strides);
        let rhs_incrs = get_strided_incrs(lhs.shape, rhs.strides);

        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let out_data = Arc::make_mut(&mut out.data);

        for (out_i, o) in out_data.iter_mut().enumerate() {
            *o = op.f(&lhs.data[lhs_i], &rhs.data[rhs_i]);

            let dim = get_incr_dim::<S>(out_i + 1, out.strides);
            incr_arg_i::<S>(&mut lhs_i, lhs_incrs, dim);
            incr_arg_i::<S>(&mut rhs_i, rhs_incrs, dim);
        }
        Ok(out)
    }
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        grad_lhs: &mut Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
        grad_rhs: &mut Self::Storage<S, E>,
        grad_out: &Self::Storage<S, E>,
    ) -> Result<(), Self::Err> {
        let lhs_incrs = get_strided_incrs(lhs.shape, lhs.strides);
        let rhs_incrs = get_strided_incrs(lhs.shape, rhs.strides);

        let mut lhs_i = 0;
        let mut rhs_i = 0;

        let lhs_data = Arc::make_mut(&mut grad_lhs.data);
        let rhs_data = Arc::make_mut(&mut grad_rhs.data);

        for out_i in 0..grad_out.data.len() {
            let go = grad_out.data[out_i];
            lhs_data[lhs_i] += op.dfdx(&lhs.data[lhs_i], &rhs.data[rhs_i]) * go;
            rhs_data[rhs_i] += op.dfdy(&lhs.data[lhs_i], &rhs.data[rhs_i]) * go;

            let dim = get_incr_dim::<S>(out_i + 1, grad_out.strides);
            incr_arg_i::<S>(&mut lhs_i, lhs_incrs, dim);
            incr_arg_i::<S>(&mut rhs_i, rhs_incrs, dim);
        }

        Ok(())
    }
}
