use crate::arrays::{Dtype, HasDtype, HasShape, Shape};

pub trait HasErr: Sized {
    type Err: std::fmt::Debug + std::fmt::Display;
}

pub trait Device: 'static + Default + Clone + HasErr {
    type Storage<S: Shape, Elem: Dtype>: 'static
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + HasShape<Shape = S>;

    fn alloc<S: Shape, E: Dtype>(&self, shape: &S) -> Result<Self::Storage<S, E>, Self::Err>;
    fn alloc_like<S: Shape, E: Dtype>(
        &self,
        storage: &Self::Storage<S, E>,
    ) -> Result<Self::Storage<S, E>, Self::Err>;
    fn random_u64(&self) -> u64;
    fn fill_with<S: Shape, E: Dtype>(&self, storage: &mut Self::Storage<S, E>, value: E);
}

pub trait HasDevice {
    type Device: Device;
}

pub trait HasDeviceStorage: HasShape + HasDtype + HasDevice {
    type Storage: 'static;
    fn dev(&self) -> &Self::Device;
    fn storage(&self) -> &Self::Storage;
    fn alloc_like(&self) -> Result<Self::Storage, <Self::Device as HasErr>::Err>;
}

pub trait Zeros<T>: HasErr {
    fn zeros(&self) -> T {
        self.try_zeros().unwrap()
    }
    fn fill_with_zeros(&self, t: &mut T);
    fn try_zeros(&self) -> Result<T, Self::Err>;
}

pub trait ZerosLike<Src, Dst>: HasErr {
    fn zeros_like(&self, src: Src) -> Dst {
        self.try_zeros_like(src).unwrap()
    }
    fn try_zeros_like(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait Ones<T>: HasErr {
    fn ones(&self) -> T {
        self.try_ones().unwrap()
    }
    fn fill_with_ones(&self, t: &mut T);
    fn try_ones(&self) -> Result<T, Self::Err>;
}

pub trait OnesLike<Src, Dst>: HasErr {
    fn ones_like(&self, src: Src) -> Dst {
        self.try_ones_like(src).unwrap()
    }
    fn try_ones_like(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait Rand<T>: HasErr {
    fn rand(&self) -> T {
        self.try_rand().unwrap()
    }
    fn fill_with_rand(&self, t: &mut T);
    fn try_rand(&self) -> Result<T, Self::Err>;
}

pub trait RandLike<Src, Dst>: HasErr {
    fn rand_like(&self, src: Src) -> Dst {
        self.try_rand_like(src).unwrap()
    }
    fn try_rand_like(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait Randn<T>: HasErr {
    fn randn(&self) -> T {
        self.try_randn().unwrap()
    }
    fn fill_with_randn(&self, t: &mut T);
    fn try_randn(&self) -> Result<T, Self::Err>;
}

pub trait RandnLike<Src, Dst>: HasErr {
    fn randn_like(&self, src: Src) -> Dst {
        self.try_randn_like(src).unwrap()
    }
    fn try_randn_like(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait TryConvert<Src, Dst>: HasErr {
    fn convert(&self, src: Src) -> Dst {
        self.try_convert(src).unwrap()
    }
    fn try_convert(&self, src: Src) -> Result<Dst, Self::Err>;
}

pub trait AsArray {
    type Array;
    fn as_array(&self) -> Self::Array;
}

pub trait AsVec {
    type Vec;
    fn as_vec(&self) -> Self::Vec;
}
