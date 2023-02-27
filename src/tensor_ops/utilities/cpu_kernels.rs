use super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    shapes::{Dtype, Shape},
    tensor::cpu::{Cpu, LendingIterator, NdIndex, StridedArray},
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
        // NOTE: we can iterate over buf here because we know inp & out
        // have exact same strides due to clone.
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

impl<E: Dtype, Op: BinaryDerivative<E>> BinaryKernel<Op, E> for Cpu {
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: &Self::Storage<S, E>,
        rhs: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err> {
        let mut out: Self::Storage<S, E> = StridedArray::new(lhs.shape)?;

        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        // NOTE: we can use buf_iter_mut() here because StridedArray::new makes a contiguous array
        for o in out.buf_iter_mut() {
            let l = lhs_iter.next().unwrap();
            let r = rhs_iter.next().unwrap();
            *o = op.f(l, r);
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
        let lhs_buf = lhs.data.as_ref();
        let rhs_buf = rhs.data.as_ref();
        let grad_lhs_buf = std::sync::Arc::make_mut(&mut grad_lhs.data);
        let grad_rhs_buf = std::sync::Arc::make_mut(&mut grad_rhs.data);
        let out_buf = grad_out.data.as_ref();

        if lhs.strides == rhs.strides && lhs.strides == grad_out.strides {
            // contiguous case
            for (i, &go) in out_buf.iter().enumerate() {
                let l = &lhs_buf[i];
                let r = &rhs_buf[i];
                grad_lhs_buf[i] += op.dfdx(l, r) * go;
                grad_rhs_buf[i] += op.dfdy(l, r) * go;
            }
        } else {
            let lhs_num_br = lhs.shape.num_elements() / lhs.data.len();
            let rhs_num_br = rhs.shape.num_elements() / rhs.data.len();

            if lhs_num_br > rhs_num_br {
                // lhs has more broadcasted - permute both shapes
                let [mut lhs_idx, mut rhs_idx, mut out_idx] =
                    index_for_binop(lhs.shape, lhs.strides, rhs.strides);
                for (l, gl) in lhs_buf.iter().zip(grad_lhs_buf.iter_mut()) {
                    let mut tmp = Default::default();
                    for _ in 0..lhs_num_br {
                        let i_rhs = rhs_idx.next().unwrap();
                        let i_out = out_idx.next().unwrap();
                        let r = &rhs_buf[i_rhs];
                        tmp += op.dfdx(l, r) * out_buf[i_out];
                        grad_rhs_buf[i_rhs] += op.dfdy(l, r) * out_buf[i_out];
                    }
                    *gl += tmp;
                }
            } else {
                // rhs has more broadcasted - permute
                let [mut rhs_idx, mut lhs_idx, mut out_idx] =
                    index_for_binop(rhs.shape, rhs.strides, lhs.strides);
                for (r, gr) in rhs_buf.iter().zip(grad_rhs_buf.iter_mut()) {
                    let mut tmp = Default::default();
                    for _ in 0..rhs_num_br {
                        let i_lhs = lhs_idx.next().unwrap();
                        let i_out = out_idx.next().unwrap();
                        let l = &lhs_buf[i_lhs];
                        tmp += op.dfdy(l, r) * out_buf[i_out];
                        grad_lhs_buf[i_lhs] += op.dfdx(l, r) * out_buf[i_out];
                    }
                    *gr += tmp;
                }
            }
        }
        Ok(())
    }
}

/// Permutes strides so that all broadcasted axes are rightmost.
#[inline(always)]
pub(crate) fn index_for_binop<S: Shape>(
    shape: S,
    strides: S::Concrete,
    other_strides: S::Concrete,
) -> [NdIndex<S>; 3] {
    let dims = shape.concrete();
    let out_strides = shape.strides();
    let mut new_shape: S::Concrete = Default::default();
    let mut new_strides: S::Concrete = Default::default();
    let mut new_other_strides: S::Concrete = Default::default();
    let mut new_out_strides: S::Concrete = Default::default();

    let mut num_broadcasted = 0;
    for d in 0..S::NUM_DIMS {
        if strides[d] == 0 {
            num_broadcasted += 1;
        }
    }
    let num_non_br = S::NUM_DIMS - num_broadcasted;

    let mut i_br = 0;
    let mut i_non_br = 0;
    for d in 0..S::NUM_DIMS {
        if strides[d] == 0 {
            // this axis is reduced
            new_shape[num_non_br + i_br] = dims[d];
            new_strides[num_non_br + i_br] = strides[d];
            new_other_strides[num_non_br + i_br] = other_strides[d];
            new_out_strides[num_non_br + i_br] = out_strides[d];
            i_br += 1;
        } else {
            new_shape[i_non_br] = dims[d];
            new_strides[i_non_br] = strides[d];
            new_other_strides[i_non_br] = other_strides[d];
            new_out_strides[i_non_br] = out_strides[d];
            i_non_br += 1;
        }
    }
    let idx = NdIndex {
        indices: Default::default(),
        shape: new_shape,
        strides: new_strides,
        next: Some(0),
        contiguous: (new_shape == dims && new_strides == shape.strides())
            .then(|| shape.num_elements()),
    };
    let other_idx = NdIndex {
        indices: Default::default(),
        shape: new_shape,
        strides: new_other_strides,
        next: Some(0),
        contiguous: (new_shape == dims && new_other_strides == shape.strides())
            .then(|| shape.num_elements()),
    };
    let out_idx = NdIndex {
        indices: Default::default(),
        shape: new_shape,
        strides: new_out_strides,
        next: Some(0),
        contiguous: (new_shape == dims && new_out_strides == shape.strides())
            .then(|| shape.num_elements()),
    };
    [idx, other_idx, out_idx]
}
