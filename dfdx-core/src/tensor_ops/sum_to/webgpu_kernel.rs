use core::any::TypeId;
use std::{sync::Arc, vec::Vec};

use wgpu::{
    BindingType, BufferBindingType, ComputePipelineDescriptor, Device, PipelineLayout, ShaderStages,
};

use crate::{
    prelude::{
        webgpu_kernels::{Forward, HasGlslType},
        Dtype, Webgpu,
    },
    tensor_ops::reduction_utils::*,
};

struct WebgpuSumKernel;

trait HasWebgpuKernel<E> {
    const MOD: &'static str;

    const FWD_SOURCE: Aligned;
    const BWD_SOURCE: Aligned;
}

#[repr(align(32))]
struct Aligned(&'static [u8]);

impl HasWebgpuKernel<f32> for Webgpu {
    const MOD: &'static str = "sum_f32";

    const FWD_SOURCE: Aligned = Aligned(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/sum_to.fwd.float.spv"
    )));
    const BWD_SOURCE: Aligned = Aligned(b"TODO");
}

impl HasWebgpuKernel<f64> for Webgpu {
    const MOD: &'static str = "sum_f32";

    const FWD_SOURCE: Aligned = Aligned(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/sum_to.fwd.double.spv"
    )));
    const BWD_SOURCE: Aligned = Aligned(b"TODO");
}

impl<E: Dtype + HasGlslType> super::SumKernel<E> for Webgpu
where
    Self: HasWebgpuKernel<E>,
{
    fn forward<Src: crate::prelude::Shape, Dst: crate::prelude::Shape, Ax: crate::prelude::Axes>(
        &self,
        dst: Dst,
        inp: &crate::prelude::Tensor<Src, E, Self>,
    ) -> Result<crate::prelude::Tensor<Dst, E, Self>, crate::prelude::Error>
    where
        Src: crate::prelude::ReduceShapeTo<Dst, Ax>,
    {
        todo!("Sum kernel has weird magic number problem");
        // TODO: Remove this, make it work with magic number
        println!(
            "{:0x}",
            u32::from_le_bytes([
                Self::FWD_SOURCE.0[0],
                Self::FWD_SOURCE.0[1],
                Self::FWD_SOURCE.0[2],
                Self::FWD_SOURCE.0[3],
            ])
        );
        if !self.shader_module_loaded(TypeId::of::<Forward<E, WebgpuSumKernel>>()) {
            self.load_shader_module::<E>(
                TypeId::of::<Forward<E, WebgpuSumKernel>>(),
                Self::FWD_SOURCE.0,
            );
        }

        let cs_module = self
            .get_shader_module(TypeId::of::<Forward<E, WebgpuSumKernel>>())
            .expect("shader module not loaded");
        let pipeline_layout = create_pipeline_layout_fwd(&self.dev);
        let pipeline = self
            .dev
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &cs_module,
                entry_point: "main",
            });

        let (dims, strides) = permute_for_reductions::<_, Ax>(inp.shape.concrete(), inp.strides);
        let num_dims = dims.len();

        let mut info = Vec::with_capacity(num_dims);
        info.extend(dims.into_iter().map(|d| d as u32));
        let dims_buffer = self.alloc_init::<u32>(&info)?;

        let mut info = Vec::with_capacity(num_dims);
        info.extend(strides.into_iter().map(|d| d as u32));
        let strides_buffer = self.alloc_init::<u32>(&info)?;

        let elems_per_thread = E::from_usize(reduction_elems_per_thread::<_, Src>(
            inp.shape.concrete(),
            inp.strides,
            Ax::as_array(),
        ))
        .unwrap();

        let physical_numel = inp.data.len::<E>();
        let physical_num_blocks = (physical_numel + 128 - 1) / 128;
        let (dst_physical_numel, dst_strides) =
            reduction_output_strides::<Ax, Src, Dst>(inp.strides, dst);
        let chunk_len = physical_numel / dst_physical_numel;

        let params_buffer = self.alloc_init::<(u32, E)>(&[(chunk_len as u32, elems_per_thread)])?;

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let storage = self.alloc_empty::<E>(dst_physical_numel)?;
        let mut entries = Vec::new();

        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Buffer(inp.data.as_entire_buffer_binding()),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::Buffer(storage.as_entire_buffer_binding()),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Buffer(params_buffer.as_entire_buffer_binding()),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::Buffer(dims_buffer.as_entire_buffer_binding()),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::Buffer(strides_buffer.as_entire_buffer_binding()),
        });

        let binding_group = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &entries,
        });
        let mut encoder = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &binding_group, &[]);
            cpass.dispatch_workgroups(physical_num_blocks as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(self.build_tensor(dst, dst_strides, storage))
    }

    fn backward<Src: crate::prelude::Shape, Dst: crate::prelude::Shape, Ax: crate::prelude::Axes>(
        &self,
        dst: Dst,
        inp: &impl crate::prelude::Tensorlike<Src, E, Self>,
        grad_inp: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), crate::prelude::Error>
    where
        Src: crate::prelude::ReduceShapeTo<Dst, Ax>,
    {
        todo!()
    }
}

fn create_pipeline_layout_fwd(dev: &Device) -> PipelineLayout {
    let entries = vec![
        // input
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // output
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // params
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // dims
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // strides
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];

    let binding_group_layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&binding_group_layout],
        push_constant_ranges: &[],
    })
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::tensor_ops::*;
    use crate::tests::*;

    #[ignore]
    #[test]
    fn test_sum_1d() {
        let dev: Webgpu = Webgpu::default();
        let t = dev.tensor([1.0, 2.0, 3.0]);
        let r = t.leaky_trace().sum::<Rank0, _>();
        let e = 6.0f64;
        assert_close_to_literal!(r, e);
        // TODO: Add exp back in
        // NOTE: .exp() to make sure its using result grad properly
        // let g = r.exp().backward();
        // assert_close_to_literal!(g.get(&t), [e.exp(); 3]);
    }
}
