use crate::arrays::{Dtype, HasShape, Shape};

pub trait Device: Default + Clone {
    type Storage<S: Shape, Elem: Dtype>: 'static
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + HasShape<Shape = S>;
    type Err: std::fmt::Debug;
    fn alloc<S: Shape, E: Dtype>(&self, shape: &S) -> Self::Storage<S, E>;
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
