use crate::{
    shapes::{RemoveDimTo, ReplaceDimTo, Shape},
    tensor::{launch_cfg, Cuda, Tensor},
};
use cudarc::driver::{DeviceSlice, LaunchAsync};

const GATHER_PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/gather.ptx"));
const SELECT_PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/select.ptx"));

macro_rules! impl_cuda_kernels {
    ($TypeName:ty, $GatherMod:tt, $GatherFwd:tt, $GatherBwd:tt, $SelectMod:tt, $SelectFwd:tt, $SelectBwd:tt) => {
        impl super::ReplaceDimKernel<$TypeName> for Cuda {
            fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
                &self,
                inp: &Tensor<Src, $TypeName, Self>,
                idx: &Tensor<Idx, usize, Self>,
            ) -> Result<Tensor<Dst, $TypeName, Self>, Self::Err>
            where
                Src: ReplaceDimTo<Dst, Idx>,
            {
                if !self.dev.has_func($GatherMod, $GatherFwd) {
                    self.dev.load_ptx(
                        GATHER_PTX_SRC.into(),
                        $GatherMod,
                        &[$GatherFwd, $GatherBwd],
                    )?;
                }

                let dst = inp.shape.replace(idx.shape);
                let numel = dst.num_elements();
                let mut storage = self.dev.alloc_zeros::<$TypeName>(numel)?;

                let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
                let idx_dims = self.dev.htod_copy(idx.shape.concrete().into())?;
                let inp_strides = self.dev.htod_copy(inp.strides.into())?;
                let idx_strides = self.dev.htod_copy(idx.strides.into())?;

                let fwd_fn = self.dev.get_func($GatherMod, $GatherFwd).unwrap();
                let cfg = launch_cfg(numel as u32);
                let params = (
                    numel,             // const size_t numel,
                    inp.data.as_ref(), // const float *inp,
                    Src::NUM_DIMS,     // const size_t inp_num_dims,
                    &inp_dims,         // const size_t *inp_dims,
                    &inp_strides,      // const size_t *inp_strides,
                    idx.data.as_ref(), // const float *idx,
                    Idx::NUM_DIMS,     // const size_t idx_num_dims,
                    &idx_dims,         // const size_t *idx_dims,
                    &idx_strides,      // const size_t *idx_strides,
                    &mut storage,      // float *out,
                    Dst::NUM_DIMS,     // const size_t out_num_dims,
                );
                unsafe { fwd_fn.launch(cfg, params) }?;

                Ok(self.build_tensor(dst, dst.strides(), storage))
            }

            fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
                &self,
                inp: &Tensor<Src, $TypeName, Self>,
                grad_inp: &mut Self::Vec<$TypeName>,
                idx: &Tensor<Idx, usize, Self>,
                _: &Tensor<Dst, $TypeName, Self>,
                grad_out: &Self::Vec<$TypeName>,
            ) -> Result<(), Self::Err>
            where
                Src: ReplaceDimTo<Dst, Idx>,
            {
                let bwd_fn = self.dev.get_func($GatherMod, $GatherBwd).unwrap();
                let numel = grad_out.len();

                let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
                let idx_dims = self.dev.htod_copy(idx.shape.concrete().into())?;
                let inp_strides = self.dev.htod_copy(inp.strides.into())?;
                let idx_strides = self.dev.htod_copy(idx.strides.into())?;

                let cfg = launch_cfg(numel as u32);
                let params = (
                    numel,             // const size_t numel,
                    grad_inp,          // float *grad_inp,
                    Src::NUM_DIMS,     // const size_t inp_num_dims,
                    &inp_dims,         // const size_t *inp_dims,
                    &inp_strides,      // const size_t *inp_strides,
                    idx.data.as_ref(), // const float *idx,
                    Idx::NUM_DIMS,     // const size_t idx_num_dims,
                    &idx_dims,         // const size_t *idx_dims,
                    &idx_strides,      // const size_t *idx_strides,
                    grad_out,          // const float *grad_out,
                    Dst::NUM_DIMS,     // const size_t out_num_dims,
                );
                unsafe { bwd_fn.launch(cfg, params) }?;
                Ok(())
            }
        }

        impl super::RemoveDimKernel<$TypeName> for Cuda {
            fn forward<Src: Shape, Dst: Shape, Idx: Shape>(
                &self,
                inp: &Tensor<Src, $TypeName, Self>,
                idx: &Tensor<Idx, usize, Self>,
            ) -> Result<Tensor<Dst, $TypeName, Self>, Self::Err>
            where
                Src: RemoveDimTo<Dst, Idx>,
            {
                if !self.dev.has_func($SelectMod, $SelectFwd) {
                    self.dev.load_ptx(
                        SELECT_PTX_SRC.into(),
                        $SelectMod,
                        &[$SelectFwd, $SelectBwd],
                    )?;
                }

                let dst = inp.shape.remove(idx.shape);
                let numel = dst.num_elements();
                let mut storage = self.dev.alloc_zeros::<$TypeName>(numel)?;

                let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
                let idx_dims = self.dev.htod_copy(idx.shape.concrete().into())?;
                let dst_dims = self.dev.htod_copy(dst.concrete().into())?;
                let inp_strides = self.dev.htod_copy(inp.strides.into())?;
                let idx_strides = self.dev.htod_copy(idx.strides.into())?;
                let dst_strides = self.dev.htod_copy(dst.strides().into())?;

                let fwd_fn = self.dev.get_func($SelectMod, $SelectFwd).unwrap();
                let cfg = launch_cfg(numel as u32);
                let params = (
                    numel,             // const size_t numel,
                    inp.data.as_ref(), // const float *inp,
                    Src::NUM_DIMS,     // const size_t inp_num_dims,
                    &inp_dims,         // const size_t *inp_dims,
                    &inp_strides,      // const size_t *inp_strides,
                    idx.data.as_ref(), // const float *idx,
                    Idx::NUM_DIMS,     // const size_t idx_num_dims,
                    &idx_dims,         // const size_t *idx_dims,
                    &idx_strides,      // const size_t *idx_strides,
                    &mut storage,      // float *out,
                    &dst_dims,         // const size_t *out_dims,
                    &dst_strides,      // const size_t *out_strides
                );
                unsafe { fwd_fn.launch(cfg, params) }?;

                Ok(self.build_tensor(dst, dst.strides(), storage))
            }

            fn backward<Src: Shape, Dst: Shape, Idx: Shape>(
                &self,
                inp: &Tensor<Src, $TypeName, Self>,
                grad_inp: &mut Self::Vec<$TypeName>,
                idx: &Tensor<Idx, usize, Self>,
                out: &Tensor<Dst, $TypeName, Self>,
                grad_out: &Self::Vec<$TypeName>,
            ) -> Result<(), Self::Err>
            where
                Src: RemoveDimTo<Dst, Idx>,
            {
                let bwd_fn = self.dev.get_func($SelectMod, $SelectBwd).unwrap();
                let numel = grad_out.len();

                let inp_dims = self.dev.htod_copy(inp.shape.concrete().into())?;
                let idx_dims = self.dev.htod_copy(idx.shape.concrete().into())?;
                let out_dims = self.dev.htod_copy(out.shape.concrete().into())?;
                let inp_strides = self.dev.htod_copy(inp.strides.into())?;
                let idx_strides = self.dev.htod_copy(idx.strides.into())?;
                let out_strides = self.dev.htod_copy(out.strides.into())?;

                let cfg = launch_cfg(numel as u32);
                let params = (
                    numel,             // const size_t numel,
                    grad_inp,          // float *grad_inp,
                    Src::NUM_DIMS,     // const size_t inp_num_dims,
                    &inp_dims,         // const size_t *inp_dims,
                    &inp_strides,      // const size_t *inp_strides,
                    idx.data.as_ref(), // const float *idx,
                    Idx::NUM_DIMS,     // const size_t idx_num_dims,
                    &idx_dims,         // const size_t *idx_dims,
                    &idx_strides,      // const size_t *idx_strides,
                    grad_out,          // const float *grad_out,
                    &out_dims,         // const size_t *out_dims,
                    &out_strides,      // const size_t *out_strides
                );
                unsafe { bwd_fn.launch(cfg, params) }?;
                Ok(())
            }
        }
    };
}

impl_cuda_kernels!(
    f32,
    "gather_f32",
    "gather_fwd_f32",
    "gather_bwd_f32",
    "select_f32",
    "select_fwd_f32",
    "select_bwd_f32"
);
impl_cuda_kernels!(
    f64,
    "gather_f64",
    "gather_fwd_f64",
    "gather_bwd_f64",
    "select_f64",
    "select_fwd_f64",
    "select_bwd_f64"
);
