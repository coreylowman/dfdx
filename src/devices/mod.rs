pub(crate) mod cpu;
pub(crate) mod device;
pub(crate) mod unary_ops;

pub(crate) use device::{BinaryKernel, FullUnaryKernel, UnaryKernel};

pub use cpu::{Cpu, CpuError};
pub use device::{
    AsArray, AsVec, Device, HasDevice, HasErr, Ones, OnesLike, Rand, RandLike, Randn, RandnLike,
    TryConvert, Zeros, ZerosLike,
};
