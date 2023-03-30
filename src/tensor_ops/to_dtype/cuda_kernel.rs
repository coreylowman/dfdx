use crate::{
    shapes::{Shape, Unit},
    tensor::{launch_cfg, unique_id, Cuda, Tensor},
};
use cudarc::{
    driver::{DeviceSlice, LaunchAsync},
    nvrtc::compile_ptx,
    types::CudaTypeName,
};
use std::sync::Arc;

const KERNEL: &str = "
#include <cstdint>
extern \"C\" __global__ void kernel(const size_t n, const $Src *inp, $Dst *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = inp[i]; }
}";

impl<E1: Unit + CudaTypeName, E2: Unit + CudaTypeName> super::ToDtypeKernel<E1, E2> for Cuda {
    fn forward<S: Shape>(inp: Tensor<S, E1, Self>) -> Result<Tensor<S, E2, Self>, Self::Err> {
        let module = std::format!("convert_{}_to_{}", E1::NAME, E2::NAME);
        let dev = &inp.device.dev;

        if !dev.has_func(&module, "kernel") {
            let src = KERNEL.replace("$Src", E1::NAME).replace("$Dst", E2::NAME);
            let ptx = compile_ptx(src).unwrap();
            dev.load_ptx(ptx, &module, &["kernel"])?;
        }

        let fwd_fn = dev.get_func(&module, "kernel").unwrap();

        let n = inp.data.len();
        let mut out = unsafe { dev.alloc::<E2>(n) }?;
        unsafe { fwd_fn.launch(launch_cfg(n as u32), (n, inp.data.as_ref(), &mut out)) }?;

        Ok(Tensor {
            id: unique_id(),
            data: Arc::new(out),
            shape: inp.shape,
            strides: inp.strides,
            device: inp.device.clone(),
            tape: Default::default(),
        })
    }
}
