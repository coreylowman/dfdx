pub(crate) mod cpu;
pub(crate) mod device;

pub(crate) use device::AllocStorageOn;

pub use cpu::{Cpu, CpuError};
pub use device::{
    AsArray, AsVec, DeviceStorage, HasErr, Ones, OnesLike, Rand, RandLike, Randn, RandnLike,
    TryConvert, Zeros, ZerosLike,
};
