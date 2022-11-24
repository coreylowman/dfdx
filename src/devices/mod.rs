pub(crate) mod cpu;
pub(crate) mod device;

pub use cpu::{Cpu, CpuError};
pub use device::{
    AsArray, AsVec, Device, HasDevice, HasDeviceStorage, HasErr, Ones, OnesLike, Rand, RandLike,
    Randn, RandnLike, TryConvert, Zeros, ZerosLike,
};
