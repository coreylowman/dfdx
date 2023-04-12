use crate::{
    shapes::{Shape, Unit},
    tensor::{launch_cfg, Cuda, Tensor},
};
use cudarc::{
    driver::{DeviceSlice, LaunchAsync},
    nvrtc::compile_ptx,
    types::CudaTypeName,
};

const KERNEL: &str = "
#if __WORDSIZE == 64
typedef long int intptr_t;
#else
typedef int intptr_t;
#endif
extern \"C\" __global__ void kernel(const size_t n, const $Src *inp, $Dst *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = inp[i]; }
}";

impl<E1: Unit + CudaTypeName, E2: Unit + CudaTypeName> super::ToDtypeKernel<E1, E2> for Cuda {
    fn forward<S: Shape>(inp: Tensor<S, E1, Self>) -> Result<Tensor<S, E2, Self>, Self::Err> {
        let module = std::format!("convert_{}_to_{}", E1::NAME, E2::NAME);
        let cuda = &inp.device;

        if !cuda.dev.has_func(&module, "kernel") {
            let src = KERNEL.replace("$Src", E1::NAME).replace("$Dst", E2::NAME);
            let ptx = compile_ptx(src).unwrap();
            cuda.dev.load_ptx(ptx, &module, &["kernel"])?;
        }

        let fwd_fn = cuda.dev.get_func(&module, "kernel").unwrap();

        let n = inp.data.len();
        let mut out = unsafe { cuda.alloc_empty::<E2>(n) }?;
        unsafe {
            fwd_fn.launch(
                launch_cfg::<128>(n as u32),
                (n, inp.data.as_ref(), &mut out),
            )
        }?;

        Ok(cuda.build_tensor(inp.shape, inp.strides, out))
    }
}
