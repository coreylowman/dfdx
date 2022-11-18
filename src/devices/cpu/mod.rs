mod allocate;
mod broadcast_reduce;
mod device;
mod index;
mod iterate;
mod kernels_binary;
mod kernels_unary;
mod matmul;
mod permute;
mod select;
// mod pool2d;

pub use device::{Cpu, CpuError};
