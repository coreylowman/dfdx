use crate::arrays::{Dtype, Shape};

pub trait HasShape {
    type Shape;
    fn shape(&self) -> &Self::Shape;
}

pub trait Device: Default + Clone {
    type Storage<S: Shape, Elem: Dtype>: std::fmt::Debug + Clone + Send + Sync + HasShape<Shape = S>;
    type Err: std::fmt::Debug;
}

pub trait Zeros<T>: Device {
    fn zeros(&self) -> T {
        self.try_zeros().unwrap()
    }
    fn fill_with_zeros(&self, t: &mut T);
    fn try_zeros(&self) -> Result<T, Self::Err>;
}

pub trait ZerosLike<Src, Dst>: Device {
    fn zeros_like(&self, src: Src) -> Dst {
        self.try_zeros_like(src).unwrap()
    }
    fn try_zeros_like(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait Ones<T>: Device {
    fn ones(&self) -> T {
        self.try_ones().unwrap()
    }
    fn fill_with_ones(&self, t: &mut T);
    fn try_ones(&self) -> Result<T, Self::Err>;
}
pub trait OnesLike<Src, Dst>: Device {
    fn ones_like(&self, src: Src) -> Dst {
        self.try_ones_like(src).unwrap()
    }
    fn try_ones_like(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait Rand<T>: Device {
    fn rand(&self) -> T {
        self.try_rand().unwrap()
    }
    fn fill_with_rand(&self, t: &mut T);
    fn try_rand(&self) -> Result<T, Self::Err>;
}

pub trait RandLike<Src, Dst>: Device {
    fn rand_like(&self, src: Src) -> Dst {
        self.try_rand_like(src).unwrap()
    }
    fn try_rand_like(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait Randn<T>: Device {
    fn randn(&self) -> T {
        self.try_randn().unwrap()
    }
    fn fill_with_randn(&self, t: &mut T);
    fn try_randn(&self) -> Result<T, Self::Err>;
}

pub trait RandnLike<Src, Dst>: Device {
    fn randn_like(&self, src: Src) -> Dst {
        self.try_randn_like(src).unwrap()
    }
    fn try_randn_like(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait TryConvert<Src, Dst>: Device {
    fn from(&self, src: Src) -> Dst {
        self.try_from(src).unwrap()
    }
    fn try_from(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait AsArray {
    type Array;
    fn as_array(&self) -> Self::Array;
}

pub trait AsVec {
    type Vec;
    fn as_vec(&self) -> Self::Vec;
}

pub trait UnaryKernel<Op, Inp: Shape, Out: Shape, Elem: Dtype>: Device {
    fn unary_fwd(
        &self,
        op: Op,
        inp: &Self::Storage<Inp, Elem>,
    ) -> Result<Self::Storage<Out, Elem>, Self::Err>;
    fn unary_bwd(
        &self,
        op: Op,
        inp: &Self::Storage<Inp, Elem>,
        grad_inp: &mut Self::Storage<Inp, Elem>,
        grad_out: &Self::Storage<Out, Elem>,
    );
}

pub trait FullUnaryKernel<Op, Inp: Shape, Out: Shape, Elem: Dtype>: Device {
    fn unary_fwd(
        &self,
        op: Op,
        inp: &Self::Storage<Inp, Elem>,
    ) -> Result<Self::Storage<Out, Elem>, Self::Err>;
    fn unary_bwd(
        &self,
        op: Op,
        inp: &Self::Storage<Inp, Elem>,
        grad_inp: &mut Self::Storage<Inp, Elem>,
        out: &Self::Storage<Out, Elem>,
        grad_out: &Self::Storage<Out, Elem>,
    );
}

pub mod unary_ops {
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
    pub struct Powi(pub(crate) i32);

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
    pub struct Select<Dst, Axis, I: super::Shape, D: super::Device> {
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
}

pub trait BinaryKernel<Op, Lhs: Shape, Rhs: Shape, Out: Shape, Elem: Dtype>: Device {
    fn binary_fwd(
        &self,
        op: Op,
        lhs: &Self::Storage<Lhs, Elem>,
        rhs: &Self::Storage<Rhs, Elem>,
    ) -> Result<Self::Storage<Out, Elem>, Self::Err>;

    fn binary_bwd(
        &self,
        op: Op,
        lhs: &Self::Storage<Lhs, Elem>,
        grad_lhs: &mut Self::Storage<Lhs, Elem>,
        rhs: &Self::Storage<Rhs, Elem>,
        grad_rhs: &mut Self::Storage<Rhs, Elem>,
        grad_out: &Self::Storage<Out, Elem>,
    );
}

pub mod binary_ops {
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
}
