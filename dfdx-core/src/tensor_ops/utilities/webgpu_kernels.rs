use crate::{
    shapes::{Dtype, Shape},
    tensor::*,
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use core::any::TypeId;
use std::{borrow::Cow, sync::Arc, vec::Vec};

pub(crate) trait UnaryOpWebgpuKernel<E> {
    const DF_USES_FX: bool;
    const HAS_CONST_DF: bool;

    /// Compiled by build.rs
    const WGSL_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .wgsl file
    const FWD_FN_NAME: &'static str;

    /// Name of function in the .wgsl file
    const BWD_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 2] = [Self::FWD_FN_NAME, Self::BWD_FN_NAME];
}

macro_rules! webgpu_unary {
    ($Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::webgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = $Wgsl;
            const MODULE_NAME: &'static str = stringify!($Op);
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (df(f(x)) $Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::webgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = true;
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = $Wgsl;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::webgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = true;
            const WGSL_SRC: &'static str = $Wgsl;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
}

pub(crate) use webgpu_unary;
use wgpu::ComputePipelineDescriptor;

impl<E: Dtype, K: UnaryOpWebgpuKernel<E> + 'static> UnaryKernel<K, E> for Webgpu {
    const BACKWARD_WITHOUT_INP: bool = K::DF_USES_FX;
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;

    fn forward<S: Shape>(
        &self,
        op: K,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        if !self.shader_module_loaded(TypeId::of::<K>()) {
            self.load_shader_module(TypeId::of::<K>(), K::WGSL_SRC);
        }

        let cs_module = self
            .get_shader_module(TypeId::of::<K>())
            .expect("shader module not loaded");
        let pipeline = self
            .dev
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &cs_module,
                entry_point: K::FWD_FN_NAME,
            });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let op_storage = self.alloc_init::<K>(&[op])?;
        let numel = inp.data.len::<E>();
        let storage = self.alloc_empty::<E>(numel)?;
        let empty = self.alloc_empty::<E>(0)?;

        match inp {
            Cow::Borrowed(inp) => {
                let binding_group = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                op_storage.as_entire_buffer_binding(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                inp.data.as_entire_buffer_binding(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                storage.as_entire_buffer_binding(),
                            ),
                        },
                    ],
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
                    cpass.dispatch_workgroups(numel as u32, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
                Ok(self.build_tensor(inp.shape, inp.strides, storage))
            }
            Cow::Owned(mut inp) => {
                let binding_group = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                op_storage.as_entire_buffer_binding(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                empty.as_entire_buffer_binding(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                Arc::make_mut(&mut inp.data).as_entire_buffer_binding(),
                            ),
                        },
                    ],
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
                    cpass.dispatch_workgroups(numel as u32, 1, 1);
                }
                self.queue.submit(Some(encoder.finish()));
                Ok(inp)
            }
        }
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        todo!()
    }
}
