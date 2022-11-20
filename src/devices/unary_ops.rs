use core::marker::PhantomData;

#[derive(Debug, Default, Clone, Copy)]
pub struct Negate;
#[derive(Debug, Default, Clone, Copy)]
pub struct Square;
#[derive(Debug, Default, Clone, Copy)]
pub struct Sqrt;
#[derive(Debug, Default, Clone, Copy)]
pub struct Tanh;
#[derive(Debug, Default, Clone, Copy)]
pub struct Sigmoid;
#[derive(Debug, Default, Clone, Copy)]
pub struct Sin;
#[derive(Debug, Default, Clone, Copy)]
pub struct Cos;
#[derive(Debug, Default, Clone, Copy)]
pub struct Ln;
#[derive(Debug, Default, Clone, Copy)]
pub struct Exp;
#[derive(Debug, Default, Clone, Copy)]
pub struct Abs;
#[derive(Debug, Default, Clone, Copy)]
pub struct ReLU;

#[derive(Debug, Clone, Copy)]
pub struct Pow<Elem>(pub(crate) Elem);

#[derive(Debug, Clone, Copy)]
pub struct NansTo<Elem>(pub(crate) Elem);

#[derive(Debug, Clone, Copy)]
pub struct Dropout {
    pub(crate) seed: u64,
    pub(crate) prob: f32,
}
#[derive(Debug, Clone, Copy)]
pub struct Clamp<Elem> {
    pub(crate) min: Elem,
    pub(crate) max: Elem,
}

#[derive(Debug, Clone, Copy)]
pub struct ScalarAdd<Elem>(pub(crate) Elem);

#[derive(Debug, Clone, Copy)]
pub struct ScalarSub<Elem>(pub(crate) Elem);

#[derive(Debug, Clone, Copy)]
pub struct ScalarMul<Elem>(pub(crate) Elem);

#[derive(Debug, Clone, Copy)]
pub struct ScalarDiv<Elem>(pub(crate) Elem);

#[derive(Debug, Default, Clone, Copy)]
pub struct Broadcast<S, Axes>(pub(crate) S, PhantomData<Axes>);
impl<S: Copy, Axes> From<&S> for Broadcast<S, Axes> {
    fn from(s: &S) -> Self {
        Self(*s, PhantomData)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Permute<S, Axes>(pub(crate) S, PhantomData<Axes>);
impl<S: Copy, Axes> From<&S> for Permute<S, Axes> {
    fn from(s: &S) -> Self {
        Self(*s, PhantomData)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Sum<Axes>(PhantomData<Axes>);
impl<Axes> From<Axes> for Sum<Axes> {
    fn from(_: Axes) -> Self {
        Self(PhantomData)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MaxReduce<Axes>(PhantomData<Axes>);
impl<Axes> From<Axes> for MaxReduce<Axes> {
    fn from(_: Axes) -> Self {
        Self(PhantomData)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MinReduce<Axes>(PhantomData<Axes>);
impl<Axes> From<Axes> for MinReduce<Axes> {
    fn from(_: Axes) -> Self {
        Self(PhantomData)
    }
}

/// TODO SelectReduce<Axis>
/// TODO SelectReplace<NewDim, Axis>
/// TODO BatchSelected<Axis>
#[derive(Debug, Clone, Copy)]
pub struct Select<Dst, Axis, I: crate::arrays::Shape, D: super::Device> {
    pub(crate) dst: Dst,
    pub(crate) indices: D::Storage<I, usize>,
    pub(crate) marker: PhantomData<Axis>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MaxPool2D<const K: usize, const S: usize, const P: usize>;

#[derive(Debug, Default, Clone, Copy)]
pub struct MinPool2D<const K: usize, const S: usize, const P: usize>;

#[derive(Debug, Default, Clone, Copy)]
pub struct AvgPool2D<const K: usize, const S: usize, const P: usize>;
