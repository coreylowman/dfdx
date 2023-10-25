use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, Error, GhostTensor, Tensor},
};
use cudarc::{
    driver::{DeviceSlice, LaunchAsync},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
    types::CudaTypeName,
};

impl<E: Dtype + CudaTypeName> super::ConcatAlongKernel<E> for Cuda {
    fn forward<A: Shape, B: Shape, C: Shape>(
        &self,
        ax: usize,
        a: &Tensor<A, E, Self>,
        b: &Tensor<B, E, Self>,
        c: &mut Tensor<C, E, Self>,
    ) -> Result<(), Error> {
        let module_name = std::format!("concat_{}", E::NAME);
        if !self.dev.has_func(&module_name, "fwd") {
            let src = KERNEL.replace("$Ty", E::NAME);
            let opts = CompileOptions {
                arch: Some(env!("CUDA_COMPUTE_CAP")),
                include_paths: vec![
                    env!("CUDA_INCLUDE_DIR").to_string(),
                    env!("OUT_DIR").to_string(),
                ],
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(src, opts).unwrap();
            self.dev.load_ptx(ptx, &module_name, &["fwd", "bwd"])?;
        }

        let fwd = self.dev.get_func(&module_name, "fwd").unwrap();
        let cfg = launch_cfg::<128>(c.data.len() as u32);

        let mut info = Vec::with_capacity(3 + 4 * A::NUM_DIMS);
        info.push(c.data.len());
        info.push(A::NUM_DIMS);
        info.push(ax);
        info.extend(a.shape.concrete());
        info.extend(a.strides);
        info.extend(b.shape.concrete());
        info.extend(b.strides);
        let info = self.dev.htod_copy(info)?;

        unsafe {
            fwd.launch(
                cfg,
                (
                    &info,
                    a.data.as_ref(),
                    b.data.as_ref(),
                    std::sync::Arc::get_mut(&mut c.data).unwrap(),
                ),
            )
        }?;

        Ok(())
    }

    fn backward<A: Shape, B: Shape>(
        &self,
        ax: usize,
        a: &GhostTensor<A, E, Self>,
        grad_a: &mut Self::Vec,
        b: &GhostTensor<B, E, Self>,
        grad_b: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        let module_name = std::format!("concat_{}", E::NAME);
        let bwd = self.dev.get_func(&module_name, "bwd").unwrap();
        let cfg = launch_cfg::<128>(grad_out.data.len() as u32);

        let mut info = Vec::with_capacity(3 + 4 * A::NUM_DIMS);
        info.push(grad_out.data.len());
        info.push(A::NUM_DIMS);
        info.push(ax);
        info.extend(a.shape.concrete());
        info.extend(a.strides);
        info.extend(b.shape.concrete());
        info.extend(b.strides);
        let info = self.dev.htod_copy(info)?;

        unsafe { bwd.launch(cfg, (&info, grad_a, grad_b, grad_out)) }?;
        Ok(())
    }
}

const KERNEL: &str = "
#include \"cuda_utils.cuh\"

extern \"C\" __global__ void fwd(
    const size_t *info, // numel, num_dims, axis, a_dims, a_strides, b_dims, b_strides
    const $Ty *lhs,
    const $Ty *rhs,
    $Ty *out
) {
    const size_t numel = info[0];
    const size_t num_dims = info[1];
    const size_t axis = info[2];
    const size_t *lhs_dims = info + 3;
    const size_t *lhs_strides = info + 3 + num_dims;
    const size_t *rhs_dims = info + 3 + 2 * num_dims;
    const size_t *rhs_strides = info + 3 + 3 * num_dims;

    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        // out_dims will be (..., lhs_dims[ax] + rhs_dims[ax], ...)
    
        // striding lhs & rhs up to the concat'd axis
        size_t i_tmp = i;
        size_t lhs_i = 0;
        size_t rhs_i = 0;
        for (int d = num_dims - 1; d > axis; d--) {
            size_t dim_i = i_tmp % lhs_dims[d];
            lhs_i += dim_i * lhs_strides[d];
            rhs_i += dim_i * rhs_strides[d];
            i_tmp /= lhs_dims[d];
        }
    
        // figure out if we are using lhs or rhs for this `i`
        size_t i_along_axis = i_tmp % (lhs_dims[axis] + rhs_dims[axis]);
        i_tmp /= (lhs_dims[axis] + rhs_dims[axis]);
    
        // striding lhs & rhs along the rest of the axes
        for (int d = axis - 1; d >= 0;d--) {
            size_t dim_i = i_tmp % lhs_dims[d];
            lhs_i += dim_i * lhs_strides[d];
            rhs_i += dim_i * rhs_strides[d];
            i_tmp /= lhs_dims[d];
        }
    
        if (i_along_axis < lhs_dims[axis]) {
            out[i] = lhs[lhs_i + i_along_axis * lhs_strides[axis]];
        } else {
            out[i] = rhs[rhs_i + (i_along_axis - lhs_dims[axis]) * rhs_strides[axis]];
        }
    }
}

extern \"C\" __global__ void bwd(
    const size_t *info, // numel, num_dims, axis, a_dims, a_strides, b_dims, b_strides
    $Ty *grad_lhs,
    $Ty *grad_rhs,
    const $Ty *grad_out
) {
    const size_t numel = info[0];
    const size_t num_dims = info[1];
    const size_t axis = info[2];
    const size_t *lhs_dims = info + 3;
    const size_t *lhs_strides = info + 3 + num_dims;
    const size_t *rhs_dims = info + 3 + 2 * num_dims;
    const size_t *rhs_strides = info + 3 + 3 * num_dims;

    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        // out_dims will be (..., lhs_dims[ax] + rhs_dims[ax], ...)
    
        // striding lhs & rhs up to the concat'd axis
        size_t i_tmp = i;
        size_t lhs_i = 0;
        size_t rhs_i = 0;
        for (int d = num_dims - 1; d > axis; d--) {
            size_t dim_i = i_tmp % lhs_dims[d];
            lhs_i += dim_i * lhs_strides[d];
            rhs_i += dim_i * rhs_strides[d];
            i_tmp /= lhs_dims[d];
        }
    
        // figure out if we are using lhs or rhs for this `i`
        size_t i_along_axis = i_tmp % (lhs_dims[axis] + rhs_dims[axis]);
        i_tmp /= (lhs_dims[axis] + rhs_dims[axis]);
    
        // striding lhs & rhs along the rest of the axes
        for (int d = axis - 1; d >= 0;d--) {
            size_t dim_i = i_tmp % lhs_dims[d];
            lhs_i += dim_i * lhs_strides[d];
            rhs_i += dim_i * rhs_strides[d];
            i_tmp /= lhs_dims[d];
        }
    
        if (i_along_axis < lhs_dims[axis]) {
            atomicAdd(grad_lhs + lhs_i + i_along_axis * lhs_strides[axis], grad_out[i]);
        } else {
            atomicAdd(grad_rhs + rhs_i + (i_along_axis - lhs_dims[axis]) * rhs_strides[axis], grad_out[i]);
        }
    }
}
";
