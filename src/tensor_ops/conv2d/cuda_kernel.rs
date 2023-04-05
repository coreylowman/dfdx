use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::cudnn::{self, Conv2dBackwardData, Conv2dBackwardFilter, Conv2dForward, CudnnDataType};
use cudarc::driver::{DeviceRepr, DeviceSlice};

use crate::{
    shapes::*,
    tensor::{launch_cfg, unique_id, Cuda, GhostTensor, Tensor},
};

use std::sync::Arc;

unsafe impl DeviceRepr for super::Conv2DOp {}

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "conv2d_f32";
    const FNS: &'static [&'static str] = &[
        "unfold_input_into_patches_f32",
        "unfold_output_into_patches_f32",
        "transpose_filters_f32",
        "sum_transposed_filters_f32",
    ];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "conv2d_f64";
    const FNS: &'static [&'static str] = &[
        "unfold_input_into_patches_f64",
        "unfold_output_into_patches_f64",
        "transpose_filters_f64",
        "sum_transposed_filters_f64",
    ];
}

fn make_4d<S: Shape>(strides: S::Concrete, pad: usize) -> [usize; 4] {
    match S::NUM_DIMS {
        3 => [pad, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => unreachable!("Only implemented for 3d & 4d arrays"),
    }
}

impl<E: Dtype + CudnnDataType> super::Conv2DKernel<E> for Cuda
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
        let conv = self.cudnn.create_conv2d::<E>(
            [op.padding as i32, op.padding as i32],
            [op.stride as i32, op.stride as i32],
            [1, 1],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        let img = self.cudnn.create_4d_tensor_ex::<E>(
            make_4d::<L>(lhs.shape.concrete(), 1).map(|x| x as i32),
            make_4d::<L>(lhs.strides, 0).map(|x| x as i32),
        )?;
        let filter = self.cudnn.create_4d_filter::<E>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            make_4d::<R>(rhs.shape.concrete(), 1).map(|x| x as i32),
        )?;
        let y = self.cudnn.create_4d_tensor_ex::<E>(
            make_4d::<O>(out.shape.concrete(), 1).map(|x| x as i32),
            make_4d::<O>(out.strides, 0).map(|x| x as i32),
        )?;
        let op = Conv2dForward {
            conv: &conv,
            x: &img,
            w: &filter,
            y: &y,
        };

        let algo = op.pick_algorithm()?;
        let workspace_size_in_bytes = op.get_workspace_size(algo)?;

        unsafe {
            let mut workspace = self.get_workspace::<u8>(workspace_size_in_bytes)?;
            let mut workspace = workspace
                .transmute_mut::<u8>(workspace_size_in_bytes)
                .unwrap();
            assert_eq!(workspace.len(), workspace_size_in_bytes);
            op.launch(
                algo,
                Some(&mut workspace),
                (E::ONE, Default::default()),
                lhs.data.as_ref(),
                rhs.data.as_ref(),
                Arc::get_mut(&mut out.data).unwrap(),
            )?;
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
        out: &GhostTensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let conv = self.cudnn.create_conv2d::<E>(
            [op.padding as i32, op.padding as i32],
            [op.stride as i32, op.stride as i32],
            [1, 1],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        let img = self.cudnn.create_4d_tensor_ex::<E>(
            make_4d::<L>(lhs.shape.concrete(), 1).map(|x| x as i32),
            make_4d::<L>(lhs.strides, 0).map(|x| x as i32),
        )?;
        let filter = self.cudnn.create_4d_filter::<E>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            make_4d::<R>(rhs.shape.concrete(), 1).map(|x| x as i32),
        )?;
        let out = self.cudnn.create_4d_tensor_ex::<E>(
            make_4d::<O>(out.shape.concrete(), 1).map(|x| x as i32),
            make_4d::<O>(out.strides, 0).map(|x| x as i32),
        )?;

        {
            let op = Conv2dBackwardData {
                conv: &conv,
                dx: &img,
                w: &filter,
                dy: &out,
            };
            let algo = op.pick_algorithm()?;
            let workspace_size_in_bytes = op.get_workspace_size(algo)?;

            unsafe {
                let mut workspace = self.get_workspace::<u8>(workspace_size_in_bytes)?;
                let mut workspace = workspace
                    .transmute_mut::<u8>(workspace_size_in_bytes)
                    .unwrap();
                assert_eq!(workspace.len(), workspace_size_in_bytes);
                op.launch(
                    algo,
                    Some(&mut workspace),
                    (E::ONE, Default::default()),
                    grad_lhs,
                    rhs.data.as_ref(),
                    grad_out,
                )
            }?;
        }

        {
            let op = Conv2dBackwardFilter {
                conv: &conv,
                x: &img,
                dw: &filter,
                dy: &out,
            };

            let algo = op.pick_algorithm()?;
            let workspace_size_in_bytes = op.get_workspace_size(algo)?;

            unsafe {
                let mut workspace = self.get_workspace::<u8>(workspace_size_in_bytes)?;
                let mut workspace = workspace
                    .transmute_mut::<u8>(workspace_size_in_bytes)
                    .unwrap();
                assert_eq!(workspace.len(), workspace_size_in_bytes);
                op.launch(
                    algo,
                    Some(&mut workspace),
                    (E::ONE, Default::default()),
                    lhs.data.as_ref(),
                    grad_rhs,
                    grad_out,
                )
            }?;
        }
        Ok(())
    }
}
