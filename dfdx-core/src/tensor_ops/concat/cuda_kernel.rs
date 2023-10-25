use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Error, Tensor},
};
use cudarc::{
    driver::{DeviceSlice, LaunchAsync},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
    types::CudaTypeName,
};

impl<E: Dtype + CudaTypeName> super::ConcatKernel<E> for Cuda {
    fn forward<A: Shape, B: Shape>(
        &self,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
    ) -> Result<Tensor<A::Catted, E, Self>, Error>
    where
        A: super::ConcatShape<B>,
    {
        debug_assert_eq!(a.strides, a.shape.strides());
        debug_assert_eq!(b.strides, b.shape.strides());
        let shape = a.shape.concat_shape(&b.shape);
        let mut buf = unsafe { self.alloc_empty::<E>(shape.num_elements()) }?;
        debug_assert_eq!(buf.len(), a.data.len() + b.data.len());
        self.dev
            .dtod_copy(a.data.as_ref(), &mut buf.slice_mut(0..a.data.len()))?;
        self.dev
            .dtod_copy(b.data.as_ref(), &mut buf.slice_mut(a.data.len()..))?;
        Ok(self.build_tensor(shape, shape.strides(), buf))
    }
    fn backward(
        &self,
        grad_a: &mut Self::Vec,
        grad_b: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let module_name = std::format!("concat_bwd_{}", E::NAME);
        if !self.dev.has_func(&module_name, "concat_bwd") {
            let src = BWD_KERNEL.replace("$Ty", E::NAME);
            let opts = CompileOptions {
                arch: Some(env!("CUDA_COMPUTE_CAP")),
                include_paths: vec![env!("CUDA_INCLUDE_DIR").to_string()],
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(src, opts).unwrap();
            self.dev.load_ptx(ptx, &module_name, &["concat_bwd"])?;
        }

        let mut offset = 0;
        {
            let f = self.dev.get_func(&module_name, "concat_bwd").unwrap();
            let numel = grad_a.len();
            let cfg = launch_cfg::<128>(numel as u32);
            unsafe { f.launch(cfg, (numel, &grad_out.slice(0..numel), grad_a)) }?;
            offset += numel;
        }
        {
            let f = self.dev.get_func(&module_name, "concat_bwd").unwrap();
            let numel = grad_b.len();
            let cfg = launch_cfg::<128>(numel as u32);
            unsafe { f.launch(cfg, (numel, &grad_out.slice(offset..), grad_b)) }?;
        }
        Ok(())
    }
}

const BWD_KERNEL: &str = "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void concat_bwd(const size_t numel, const $Ty *inp, $Ty *out) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] += inp[i];
    }
}
";
