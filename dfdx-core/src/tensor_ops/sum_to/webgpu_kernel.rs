use core::any::TypeId;

use wgpu::ComputePipelineDescriptor;

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
    const FNS: &'static [&'static str];
}

impl HasWebgpuKernel<f32> for Webgpu {
    const MOD: &'static str = "sum_f32";
    const FNS: &'static [&'static str] = &["sum_to_fwd_f32", "sum_to_bwd_f32"];
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
        if !self.shader_module_loaded(TypeId::of::<Forward<E, WebgpuSumKernel>>()) {
            self.load_shader_module::<E>(
                TypeId::of::<Forward<E, WebgpuSumKernel>>(),
                include_bytes!(concat!(env!("OUT_DIR"), "/sum_to.fwd.float.spv")),
            );
        }

        let cs_module = self
            .get_shader_module(TypeId::of::<Forward<E, WebgpuSumKernel>>())
            .expect("shader module not loaded");
        let pipeline = self
            .dev
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &cs_module,
                entry_point: "main",
            });

        let (dims, strides) = permute_for_reductions::<_, Ax>(inp.shape.concrete(), inp.strides);
        let num_dims = dims.len();

        let mut info = Vec::with_capacity(num_dims * 2);
        info.extend(dims);
        info.extend(strides);
        let info_buffer = self.alloc_empty::<u32>(num_dims * 2)?;
        info_buffer.copy_to_device(&self.dev, &self.queue, &info);

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

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let storage = self.alloc_empty::<E>(dst_physical_numel)?;
        let mut entries = Vec::new();

        todo!("add buffers to entries, but we need to get atomic operations working");

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
