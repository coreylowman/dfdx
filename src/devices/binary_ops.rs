use crate::arrays::Dtype;

#[derive(Debug, Default, Clone, Copy)]
pub struct Add;
#[derive(Debug, Default, Clone, Copy)]
pub struct Sub;
#[derive(Debug, Default, Clone, Copy)]
pub struct Mul;
#[derive(Debug, Default, Clone, Copy)]
pub struct Div;
#[derive(Debug, Default, Clone, Copy)]
pub struct MinBinary;
#[derive(Debug, Default, Clone, Copy)]
pub struct MaxBinary;

#[derive(Debug, Default, Clone, Copy)]
pub struct MatMul;

#[derive(Debug, Default, Clone, Copy)]
pub struct Conv2D<const K: usize, const S: usize, const P: usize>;

#[derive(Debug, Default, Clone, Copy)]
pub struct HuberError<E: Dtype> {
    pub(crate) delta: E,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BCEWithLogits;
