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

impl<E: Dtype + CudaTypeName> super::ReshapeKernel<E> for Cuda {
    fn forward<Src: Shape, Dst: Shape>(
        &self,
        dst: &Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err> {
        let module = std::format!("reshape_fwd_{}", E::NAME);
        if !self.dev.has_func(&module, "reshape_fwd") {
            let src = FWD_KERNEL.replace("$T", E::NAME);
            let opts = CompileOptions {
                arch: Some(env!("CUDA_COMPUTE_CAP")),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(src, opts).unwrap();
            self.dev.load_ptx(ptx, &module, &["reshape_fwd"])?;
        }
        let fwd_fn = self.dev.get_func(&module, "reshape_fwd").unwrap();

        let numel = inp.shape.num_elements();
        let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;

        let mut info = Vec::with_capacity(Src::NUM_DIMS * 2 + Dst::NUM_DIMS * 2);
        info.extend(inp.shape.concrete());
        info.extend(inp.strides);
        info.extend(dst.concrete());
        info.extend(dst.strides());
        let info = self.dev.htod_copy(info)?;

        let cfg = launch_cfg::<128>(numel as u32);
        let params = (
            numel,             // const size_t numel,
            Src::NUM_DIMS,     // const size_t inp_num_dims,
            Dst::NUM_DIMS,     // const size_t out_num_dims,
            &info,             // const size_t *info,
            inp.data.as_ref(), // const float *inp,
            &mut storage,      // float *out
        );
        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(self.build_tensor(*dst, dst.strides(), storage))
    }

    fn backward<Src: Shape, Dst: Shape>(
        &self,
        dst: &Dst,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let module = std::format!("reshape_bwd_{}", E::NAME);
        if !self.dev.has_func(&module, "reshape_bwd") {
            let src = BWD_KERNEL.replace("$T", E::NAME);
            let opts = CompileOptions {
                arch: Some(env!("CUDA_COMPUTE_CAP")),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(src, opts).unwrap();
            self.dev.load_ptx(ptx, &module, &["reshape_bwd"])?;
        }
        let bwd_fn = self.dev.get_func(&module, "reshape_bwd").unwrap();

        let numel = grad_inp.len();

        let mut info = Vec::with_capacity(Src::NUM_DIMS * 2 + Dst::NUM_DIMS * 2);
        info.extend(inp.shape.concrete());
        info.extend(inp.strides);
        info.extend(dst.concrete());
        info.extend(dst.strides());
        let info = self.dev.htod_copy(info)?;

        let cfg = launch_cfg::<128>(numel as u32);
        let params = (
            numel,         // const size_t numel,
            Src::NUM_DIMS, // const size_t inp_num_dims,
            Dst::NUM_DIMS, // const size_t out_num_dims,
            &info,         // const size_t *info,
            grad_inp,      // float *grad_inp,
            grad_out,      // const float *grad_out,
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}

const FWD_KERNEL: &str = "
#if __WORDSIZE == 64
typedef long int intptr_t;
#else
typedef int intptr_t;
#endif

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

extern \"C\" __global__ void reshape_fwd(
    const size_t numel,
    const size_t inp_num_dims,
    const size_t out_num_dims,
    const size_t *info,
    const $T *inp,
    $T *out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    const size_t *inp_dims = info;
    const size_t *inp_strides = info + inp_num_dims;
    const size_t *out_dims = info + 2 * inp_num_dims;
    const size_t *out_strides = info + 2 * inp_num_dims + out_num_dims;

    unsigned int inp_i = get_strided_index(i, inp_num_dims, inp_dims, inp_strides);
    unsigned int out_i = get_strided_index(i, out_num_dims, out_dims, out_strides);

    out[out_i] = inp[inp_i];
}
";

const BWD_KERNEL: &str = "
#if __WORDSIZE == 64
typedef long int intptr_t;
#else
typedef int intptr_t;
#endif

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

extern \"C\" __global__ void reshape_bwd(
    const size_t numel,
    const size_t inp_num_dims,
    const size_t out_num_dims,
    const size_t *info,
    $T *grad_inp,
    const $T *grad_out
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }

    const size_t *inp_dims = info;
    const size_t *inp_strides = info + inp_num_dims;
    const size_t *out_dims = info + 2 * inp_num_dims;
    const size_t *out_strides = info + 2 * inp_num_dims + out_num_dims;

    unsigned int inp_i = get_strided_index(i, inp_num_dims, inp_dims, inp_strides);
    unsigned int out_i = get_strided_index(i, out_num_dims, out_dims, out_strides);

    atomicAdd(grad_inp + inp_i, grad_out[out_i]);
}
";
