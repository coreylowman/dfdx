mod allocate;
mod device;

pub use device::{Cuda, CudaError};

pub(crate) fn launch_cfg(n: u32) -> cudarc::driver::LaunchConfig {
    const NUM_THREADS: u32 = 128;
    let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}
