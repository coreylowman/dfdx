use crate::{
    prelude::Storage,
    tensor::webgpu::{Webgpu, WebgpuNativeType},
    tensor_ops::utilities::webgpu_kernels::webgpu_params,
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
        let label = Some(module_name.as_str());
        let device = inp.device;

        let layout = device.dev.create_bind_group_layout(&LAYOUT_DESC);
        let shader_source: String = KERNEL
            .replace("__SRC__", E1::NAME)
            .replace("__DST__", E2::NAME);

        // TODO: support WGSL shaders in device shader cache
        let source = wgpu::ShaderSource::Wgsl(shader_source.into());
        let shader_module = device.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_name),
            source,
        });
        let pipeline_layout = device
            .dev
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: label.clone(),
                bind_group_layouts: layouts,
                // todo: these are useful and we should use them if the adapter supports them
                push_constant_ranges: &push_constant_ranges,
            });

        let pipeline = device
            .dev
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: label.clone(),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: fn_name,
            });

        let numel = inp.shape.num_elements();
        let work_groups: (u32, u32, u32) = (numel as u32, 1, 1);
        let shape = inp.shape;
        let strides = shape.strides();
        let output = unsafe { device.alloc_empty::<E2>(numel) }?;

        let params: wgpu::BindGroup = webgpu_params!(device, pipeline; inp.data, output);

        let _idx = device.submit_commands(label.clone(), |encoder| {
            let (x, y, z) = *work_groups;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: label.clone(),
                ..Default::default()
            });
            // TODO: should this be called before the pass, as the pass is created, or before submission?
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &params, &[]);
            pass.dispatch_workgroups(numel as u32, 1, 1);
        });

        // note: no need to sync here, buffer can remain on the gpu until to_array or to_vec gets called,
        // and those functions sync the device before mapping the buffer
        Ok(device.build_tensor(shape, strides, output))
    }
}
