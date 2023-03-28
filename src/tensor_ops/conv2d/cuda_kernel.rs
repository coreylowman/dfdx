use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::driver::{DeviceRepr, LaunchAsync, ValidAsZeroBits};
use cudarc::nvrtc::compile_ptx;

use crate::{
    shapes::*,
    tensor::{launch_cfg, unique_id, Cuda, Tensor},
};

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

unsafe impl DeviceRepr for super::Conv2DOp {}

trait HasCudaKernel<E> {
    const TYPENAME: &'static str;
}

impl HasCudaKernel<f32> for Cuda {
    const TYPENAME: &'static str = "float";
}

impl HasCudaKernel<f64> for Cuda {
    const TYPENAME: &'static str = "double";
}

fn mod_name(op: &super::Conv2DOp) -> std::string::String {
    let mut hasher = DefaultHasher::new();
    op.hash(&mut hasher);
    let hash = hasher.finish();
    std::format!("conv2d_{hash}")
}

fn make_4d<S: Shape>(strides: S::Concrete) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [0, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => unreachable!("Only implemented for 3d & 4d arrays"),
    }
}

impl<E: Dtype + ValidAsZeroBits> super::Conv2DKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
    CudaBlas: Gemm<E>,
{
    fn alloc<S: Shape>(&self, shape: S) -> Result<Tensor<S, E, Self>, Self::Err> {
        let data = Arc::new(unsafe { self.dev.alloc::<E>(shape.num_elements()) }?);
        Ok(Tensor {
            id: unique_id(),
            data,
            shape,
            strides: shape.strides(),
            device: self.clone(),
            tape: Default::default(),
        })
    }
    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::Conv2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let name = mod_name(&op);
        if !self.dev.has_func(&name, "unfold_input") {
            let src = KERNEL_SRC;
            let src = src.replace("op.stride", &op.stride.to_string());
            let src = src.replace("op.padding", &op.padding.to_string());
            let src = src.replace("op.kernel", &op.kernel.to_string());
            let src = src.replace("op.batch", &op.batch.to_string());
            let src = src.replace("op.chan_in", &op.chan_in.to_string());
            let src = src.replace("op.chan_out", &op.chan_out.to_string());
            let src = src.replace("op.h_in", &op.h_in.to_string());
            let src = src.replace("op.h_out", &op.h_out.to_string());
            let src = src.replace("op.w_in", &op.w_in.to_string());
            let src = src.replace("op.w_out", &op.w_out.to_string());
            let src = src.replace("$TY", Self::TYPENAME);
            let ptx = compile_ptx(src).unwrap();
            self.dev.load_ptx(
                ptx,
                &name,
                &[
                    "unfold_input",
                    "unfold_output",
                    "transpose_filters",
                    "sum_transposed_filters",
                ],
            )?;
        }

        let patches_item_numel = op.chan_in * op.kernel * op.kernel * op.h_out * op.w_out;
        let patches_numel = op.batch * patches_item_numel;

        let mut patches = unsafe { self.get_workspace::<E>(patches_numel) }?;
        let mut patches = unsafe { patches.transmute_mut::<E>(patches_numel).unwrap() };

        let img_strides = self.dev.htod_copy(make_4d::<L>(lhs.strides).into())?;
        let unfold_fn = self.dev.get_func(&name, "unfold_input").unwrap();
        let cfg = launch_cfg((op.batch * op.chan_in * op.h_out * op.w_out) as u32);
        let params = (lhs.data.as_ref(), &img_strides, &mut patches);
        unsafe { unfold_fn.launch(cfg, params) }?;

        // (O, C * K * K) * (B, C * K * K, OH * OW) = (B, O, OH * OW)
        let m = op.chan_out;
        let k = op.chan_in * op.kernel * op.kernel;
        let n = op.h_out * op.w_out;
        unsafe {
            self.gemm_batch(
                (op.batch, m, k, n),
                rhs.data.as_ref(),
                [0, k, 1],
                &patches,
                [k * n, n, 1],
                Default::default(),
                Arc::get_mut(&mut out.data).unwrap(),
                [m * n, n, 1],
            )
            .unwrap();
        }

        Ok(())
    }

    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::Conv2DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        _: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let name = mod_name(&op);
        let patches_item_numel = op.chan_out * op.kernel * op.kernel * op.h_in * op.w_in;
        let patches_numel = op.batch * patches_item_numel;
        let filters_numel = op.chan_in * op.chan_out * op.kernel * op.kernel;

        let mut patches = unsafe { self.get_workspace::<E>(patches_numel) }?;
        let mut patches = unsafe { patches.transmute_mut::<E>(patches_numel).unwrap() };

        let mut f_b1023 = unsafe { self.dev.alloc::<E>(filters_numel) }?;
        let mut grad_f_b1023 = unsafe { self.dev.alloc::<E>(op.batch * filters_numel) }?;
        let f_strides = self.dev.htod_copy(rhs.strides.into())?;

        self.par_stream.wait_for_default()?;

        {
            // unfold grad_out into patches
            let unfold_fn = self.dev.get_func(&name, "unfold_output").unwrap();
            let cfg = launch_cfg((op.batch * op.chan_out * op.h_in * op.w_in) as u32);
            unsafe { unfold_fn.launch(cfg, (grad_out, &mut patches)) }?;
        }

        {
            // prepare filters for backward operations by
            // swapping dims 0 and 1
            let tr_fn = self.dev.get_func(&name, "transpose_filters").unwrap();
            let cfg = launch_cfg(rhs.shape.num_elements() as u32);
            unsafe {
                tr_fn.launch_on_stream(
                    self.par_stream.as_ref(),
                    cfg,
                    (rhs.data.as_ref(), &f_strides, &mut f_b1023),
                )
            }?;

            self.par_stream.wait_for_default()?;

            // img_g += filters * patches
            // (B, C, H * W) += (B, C, O * K * K) * (B, O * K * K, H * W)
            let m = op.chan_in;
            let k = op.chan_out * op.kernel * op.kernel;
            let n = op.h_in * op.w_in;
            unsafe {
                self.blas.set_stream(Some(self.par_stream.as_ref()))?;
                self.gemm_batch(
                    (op.batch, m, k, n),
                    &f_b1023,
                    [0, k, 1],
                    &patches,
                    [k * n, n, 1],
                    <E>::ONE,
                    grad_lhs,
                    [m * n, n, 1],
                )
                .unwrap();
                self.blas.set_stream(None)?;
            }
        }

        {
            // weight_g += img * patches^T
            // (B, C, O * K * K) += (B, C, H * W) * (B, H * W, O * K * K)
            let m = op.chan_in;
            let k = op.h_in * op.w_in;
            let n = op.chan_out * op.kernel * op.kernel;
            unsafe {
                self.gemm_batch(
                    (op.batch, m, k, n),
                    lhs.data.as_ref(),
                    [m * k, k, 1],
                    &patches,
                    [k * n, 1, k],
                    Default::default(),
                    &mut grad_f_b1023,
                    [m * n, n, 1],
                )
                .unwrap();
            }

            // sum all the gradients collected in our broadcasted grad_f
            // into grad_rhs
            let sum_fn = self.dev.get_func(&name, "sum_transposed_filters").unwrap();
            let cfg = launch_cfg(rhs.shape.num_elements() as u32);
            unsafe { sum_fn.launch(cfg, (&grad_f_b1023, grad_rhs, &f_strides)) }?;
        }

        self.dev.wait_for(self.par_stream.as_ref())?;

        Ok(())
    }
}

const KERNEL_SRC: &str = "
extern \"C\" __global__ void unfold_input(
    const $TY *image, // 4d (Batch, Channels, Height, Width)
    const size_t *strides, // 4d image strides
    $TY *patches // 6d (Batch, Channels, KernelSize, KernelSize, HeightOut, WidthOut)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t item_numel = op.batch * op.chan_in * op.h_out * op.w_out;
    if (i >= item_numel) {
        return;
    }

    // patches shape is (B, C, K, K, h_out, w_out)
    unsigned int idx = i;
    const size_t ow = idx % op.w_out;
    idx /= op.w_out;
    const size_t oh = idx % op.h_out;
    idx /= op.h_out;
    const size_t c = idx % op.chan_in;
    idx /= op.chan_in;
    const size_t b = idx % op.batch;

    $TY *patches_ptr = patches + b * (op.chan_in * op.kernel * op.kernel * op.h_out * op.w_out) + c * (op.kernel * op.kernel * op.h_out * op.w_out) + oh * op.w_out + ow;
    const $TY *img_ptr = image + b * strides[0] + c * strides[1];

    for (int k1 = 0;k1 < op.kernel;k1++) {
        for (int k2 = 0;k2 < op.kernel;k2++) {
            const size_t y_plus_p = oh * op.stride + k1;
            const size_t y = y_plus_p - op.padding;
            const size_t x_plus_p = ow * op.stride + k2;
            const size_t x = x_plus_p - op.padding;
            *patches_ptr = (y >= op.h_in || x >= op.w_in) ? 0.0 : img_ptr[y * strides[2] + x * strides[3]];
            patches_ptr += op.h_out * op.w_out;
        }
    }
}

extern \"C\" __global__ void unfold_output(
    const $TY *image_out, // 4d (Batch, ChanOut, HeightOut, WidthOut)
    $TY *patches // 6d (Batch, ChanOut, KernelSize, KernelSize, HeightIn, WidthIn)
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t item_numel = op.batch * op.chan_out * op.h_in * op.w_in;
    if (i >= item_numel) {
        return;
    }

    unsigned int idx = i;
    const size_t x = idx % op.w_in;
    idx /= op.w_in;
    const size_t y = idx % op.h_in;
    idx /= op.h_in;
    const size_t o = idx % op.chan_out;
    idx /= op.chan_out;
    const size_t b = idx % op.batch;

    $TY *patches_ptr = patches + b * (op.chan_out * op.kernel * op.kernel * op.h_in * op.w_in) + o * (op.kernel * op.kernel * op.h_in * op.w_in) + y * op.w_in + x;
    const $TY *img_ptr = image_out + b * (op.chan_out * op.h_out * op.w_out) + o * (op.h_out * op.w_out);

    for (int k1 = 0;k1 < op.kernel;k1++) {
        for (int k2 = 0;k2 < op.kernel;k2++) {
            const size_t oh_ks = y + op.padding;
            const size_t oh_s = oh_ks - k1;
            const size_t oh = oh_s / op.stride;
            const size_t ow_ks = x + op.padding;
            const size_t ow_s = ow_ks - k2;
            const size_t ow = ow_s / op.stride;
        
            const bool invalid = (oh_ks < k1 || oh_s % op.stride != 0 || oh >= op.h_out)
                || (ow_ks < k2 || ow_s % op.stride != 0 || ow >= op.w_out);

            *patches_ptr = invalid ? 0.0 : img_ptr[oh * op.w_out  + ow];
            patches_ptr += op.h_in * op.w_in;
        }
    }
}

extern \"C\" __global__ void transpose_filters(
    const $TY *filters, // 4d (ChanOut, ChanIn, KernelSize, KernelSize)
    const size_t *strides, // 4d filters strides
    $TY *filters_tr // 4d (ChanIn, ChanOut, KernelSize, KernelSize)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = op.chan_in * op.chan_out * op.kernel * op.kernel;
    if (i >= numel) {
        return;
    }

    unsigned int idx = i;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t c = idx % op.chan_in;
    idx /= op.chan_in;
    const size_t o = idx % op.chan_out;

    size_t i_tr = c * (op.chan_out * op.kernel * op.kernel) + o * (op.kernel * op.kernel) + k1 * (op.kernel) + k2;
    size_t i_no = o * strides[0] + c * strides[1] + k1 * strides[2] + k2 * strides[3];

    filters_tr[i_tr] = filters[i_no];
}

extern \"C\" __global__ void sum_transposed_filters(
    const $TY *filters_tr, // 5d (Batch, ChanIn, ChanOut, KernelSize, KernelSize)
    $TY *filters, // 4d (ChanOut, ChanIn, KernelSize, KernelSize)
    const size_t *strides // 4d filter strides
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = op.chan_out * op.chan_in * op.kernel * op.kernel;
    if (i >= numel) {
        return;
    }

    unsigned int idx = i;
    const size_t k2 = idx % op.kernel;
    idx /= op.kernel;
    const size_t k1 = idx % op.kernel;
    idx /= op.kernel;
    const size_t c = idx % op.chan_in;
    idx /= op.chan_in;
    const size_t o = idx % op.chan_out;

    size_t i_tr = c * (op.chan_out * op.kernel * op.kernel) + o * (op.kernel * op.kernel) + k1 * (op.kernel) + k2;
    size_t i_no = o * strides[0] + c * strides[1] + k1 * strides[2] + k2 * strides[3];

    $TY tmp = 0.0;
    for (int b = 0; b < op.batch; b++) {
        tmp += filters_tr[b * numel + i_tr];
    }

    filters[i_no] += tmp;
}
";
