use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Tensor},
};
use cudarc::{
    driver::{DeviceSlice, LaunchAsync},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
    types::CudaTypeName,
};
use std::vec::Vec;

impl<E: Dtype + CudaTypeName> super::StackKernel<E> for Cuda {
    fn forward<S: Shape, Num: Dim>(
        &self,
        num: Num,
        inps: &[Tensor<S, E, Self>],
    ) -> Result<Tensor<S::Larger, E, Self>, Self::Err>
    where
        S: super::AddDim<Num>,
    {
        debug_assert_eq!(inps.len(), num.size());

        // check that all the strides are the same
        let item_strides = inps[0].strides;
        for i in inps.iter() {
            assert_eq!(i.strides, item_strides);
        }
        let shape: S::Larger = inps[0].shape().add_dim(num);

        // build the new strides
        let mut strides = shape.strides();
        strides[0] = inps[0].data.len();
        for d in 1..<S::Larger as Shape>::NUM_DIMS {
            strides[d] = item_strides[d - 1];
        }

        // copy the data
        let item_numel = strides[0];
        let mut data = unsafe { self.alloc_empty::<E>(num.size() * item_numel) }?;
        let mut offset = 0;
        for item in inps {
            debug_assert_eq!(item.data.len(), item_numel);
            self.dev.dtod_copy(
                item.data.as_ref(),
                &mut data.slice_mut(offset..offset + item_numel),
            )?;
            offset += item_numel;
        }
        debug_assert_eq!(offset, data.len());
        Ok(self.build_tensor(shape, strides, data))
    }

    fn backward(
        &self,
        mut grad_inp: Vec<&mut Self::Vec<E>>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let module_name = std::format!("stack_bwd_{}", E::NAME);
        if !self.dev.has_func(&module_name, "stack_bwd") {
            let src = BWD_KERNEL.replace("$Ty", E::NAME);
            let opts = CompileOptions {
                arch: Some(env!("CUDA_COMPUTE_CAP")),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(src, opts).unwrap();
            self.dev.load_ptx(ptx, &module_name, &["stack_bwd"])?;
        }

        let mut offset = 0;
        for item in grad_inp.drain(..) {
            let f = self.dev.get_func(&module_name, "stack_bwd").unwrap();
            let numel: usize = item.len();
            let cfg = launch_cfg::<128>(numel as u32);
            let sub = grad_out.slice(offset..offset + numel);
            unsafe { f.launch(cfg, (numel, &sub, item)) }?;
            offset += numel;
        }
        debug_assert_eq!(offset, grad_out.len());
        Ok(())
    }
}

const BWD_KERNEL: &str = "
extern \"C\" __global__ void stack_bwd(const size_t numel, const $Ty *inp, $Ty *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) { out[i] += inp[i]; }
}
";
