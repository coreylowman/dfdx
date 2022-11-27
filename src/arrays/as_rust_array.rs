use super::*;
use std::vec::Vec;

pub trait RustArrayRepr<E: Dtype>: Shape {
    type Repr: Clone + std::fmt::Debug + Send + Sync;
}

impl<E: Dtype> RustArrayRepr<E> for Rank0 {
    type Repr = E;
}

impl<const M: usize, E: Dtype> RustArrayRepr<E> for Rank1<M> {
    type Repr = [E; M];
}

impl<const M: usize, const N: usize, E: Dtype> RustArrayRepr<E> for Rank2<M, N> {
    type Repr = [[E; N]; M];
}

impl<const M: usize, const N: usize, const O: usize, E: Dtype> RustArrayRepr<E> for Rank3<M, N, O> {
    type Repr = [[[E; O]; N]; M];
}

impl<const M: usize, const N: usize, const O: usize, const P: usize, E: Dtype> RustArrayRepr<E>
    for Rank4<M, N, O, P>
{
    type Repr = [[[[E; P]; O]; N]; M];
}

impl<E: Dtype> RustArrayRepr<E> for (Dyn,) {
    type Repr = Vec<E>;
}

impl<E: Dtype> RustArrayRepr<E> for (Dyn, Dyn) {
    type Repr = Vec<E>;
}

impl<E: Dtype, const M: usize> RustArrayRepr<E> for (Dyn, Const<M>) {
    type Repr = Vec<E>;
}

impl<E: Dtype, const M: usize> RustArrayRepr<E> for (Const<M>, Dyn) {
    type Repr = Vec<E>;
}
