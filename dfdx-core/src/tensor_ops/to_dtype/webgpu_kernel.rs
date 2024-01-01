use crate::{
    tensor::webgpu::{Webgpu, WebgpuNativeType},
    tensor_ops::utilities::webgpu_kernels::webgpu_params, prelude::Storage
};
use num_traits::AsPrimitive;
use wgpu;

/// kernel template
const KERNEL: &'static str = include_str!("./to_dtype.wgsl");

const LAYOUT_DESC: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
    label: Some("to-dtype"),
    entries: &[
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ],
};

impl<E1: WebgpuNativeType + AsPrimitive<E2>, E2: WebgpuNativeType> super::ToDtypeKernel<E1, E2>
    for Webgpu
{
    fn forward<S: crate::prelude::Shape>(
        inp: crate::prelude::Tensor<S, E1, Self>,
    ) -> Result<crate::prelude::Tensor<S, E2, Self>, crate::prelude::Error> {
        let module_name = std::format!("convert_{}_to_{}", E1::NAME, E2::NAME);
        let device = inp.device;

        let layout = device.dev.create_bind_group_layout(&LAYOUT_DESC);
        let shader_source: String = KERNEL
            .replace("__SRC__", E1::NAME)
            .replace("__DST__", E2::NAME);

        let pipeline = device.load_pipeline(
            module_name.as_str(),
            shader_source.as_str(),
            "main",
            &[&layout],
        );

        let numel = inp.shape.num_elements();
        let work_groups: (u32, u32, u32) = (numel as u32, 1, 1);
        let shape = inp.shape;
        let strides = shape.strides();
        let output = unsafe { device.alloc_empty::<E2>(numel) }?;

        let params: wgpu::BindGroup = webgpu_params!(device, pipeline; inp.data, output);

        let _idx  = device.submit_basic_op(&pipeline, &params, Some(module_name.as_str()), &work_groups);

        Ok(device.build_tensor(shape, strides, output))
    }
}
