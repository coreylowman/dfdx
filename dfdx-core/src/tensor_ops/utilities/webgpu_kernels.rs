extern crate alloc;

use crate::{
    shapes::{Dtype, Shape},
    tensor::{webgpu::Webgpu, Error, Tensor},
    tensor_ops::ops::{UnaryKernel, BinaryKernel},
};

use alloc::{borrow::Cow, sync::Arc};

/// Creates a [`BindGroup`] for a pipeline from a set of [`wgpu::BindingResource`]s.
macro_rules! webgpu_params {
    ($self:expr, $pipeline:expr; $($x:expr),+ $(,)? ) => {
        {
            let bindings = [$($x.as_entire_binding()),+];
            let entries: Vec<_> = bindings
                .into_iter()
                .enumerate()
                .map(|(i, binding)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: binding,
                })
                .collect();
            $self.dev.create_bind_group(&::wgpu::BindGroupDescriptor {
                label: None,
                layout: &($pipeline).get_bind_group_layout(0),
                entries: &entries
            })
        }
    }
}
pub(crate) use webgpu_params;

pub trait UnaryOpWebgpuKernel<E> {
    const DF_USES_FX: bool;
    const HAS_CONST_DF: bool;

    /// WGSL source code for the kernel
    const WGSL_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .wgsl file (used as entrypoint)
    const FWD_FN_NAME: &'static str;

    /// Name of function in the .ggsl file (used as entrypoint)
    const BWD_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 2] = [Self::FWD_FN_NAME, Self::BWD_FN_NAME];
}

macro_rules! webgpu_unary {
    ($Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::wgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = include_str!($Wgsl);
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (df(f(x)) $Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::wgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = true;
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = include_str!($Wgsl);
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::wgpu_kernels::UnaryOpWebgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = true;
            const WGSL_SRC: &'static str = include_str!($Wgsl);
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
}
pub(crate) use webgpu_unary;
impl<E: Dtype, K: UnaryOpWebgpuKernel<E>> UnaryKernel<K, E> for Webgpu {
    const BACKWARD_WITHOUT_INP: bool = K::DF_USES_FX;
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;

    fn forward<S: Shape>(
        &self,
        op: K,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        todo!("Webgpu unary forwards")
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        inp: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        todo!("Wgpu unary backwards")
    }
}


pub trait BinaryOpWebgpuKernel<E> {
    const HAS_CONST_DF: bool;

    /// WGSL source code for the kernel
    const WGSL_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .wgsl file
    const FWD_FN_NAME: &'static str;

    /// Name of function in the .wgsl file
    const BWD_LHS_FN_NAME: &'static str;

    /// Name of function in the .wgsl file
    const BWD_RHS_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 3] = [
        Self::FWD_FN_NAME,
        Self::BWD_LHS_FN_NAME,
        Self::BWD_RHS_FN_NAME,
    ];
}
macro_rules! wgpu_binary {
    ($Op:path, $TypeName:ty, $Wgsl:tt, $Mod:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::webgpu_kernels::BinaryOpWebgpuKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = $Wgsl;
            const MODULE_NAME: &'static str = $Mod;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Wgsl:tt, $Mod:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::webgpu_kernels::BinaryOpWebgpuKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = true;
            const WGSL_SRC: &'static str = $Wgsl;
            const MODULE_NAME: &'static str = $Mod;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
}

pub(crate) use wgpu_binary;

impl<E: Dtype, K: BinaryOpWebgpuKernel<E> + Clone> BinaryKernel<K, E> for Webgpu {
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: K,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Error> {
        let shape = match &lhs {
            Cow::Borrowed(lhs) => lhs.shape,
            Cow::Owned(lhs) => lhs.shape,
        };
        let strides = shape.strides();
        let numel = shape.num_elements();
        // todo: dream about memory64
        // https://github.com/WebAssembly/memory64
        let work_groups: (u32, u32, u32) = (numel as u32, 1, 1);

        // todo: pipeline caching
        let fwd_pipeline = self.load_binary_pipeline(K::MODULE_NAME, K::WGSL_SRC, K::FWD_FN_NAME);


        let output = unsafe { self.alloc_empty::<E>(numel) }?;

        // note: storage buffers cannot be both read from and written to within
        // the same pipeline stage, so Cow doesn't change operation behavior.
        {
            // let (lhs, rhs) = (&lhs, &rhs);
            let lhs: &Tensor<S, E, Self> = lhs.as_ref();
            let rhs: &Tensor<S, E, Self> = rhs.as_ref();
            let params: wgpu::BindGroup = webgpu_params!(self, fwd_pipeline; lhs.data, rhs.data, output);
            let _idx = self.submit_basic_op(&fwd_pipeline, &params, Some(K::FWD_FN_NAME), &work_groups);
        }
        Ok(self.build_tensor(shape, strides, output))
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        lhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl crate::prelude::Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Error> {
        todo!("Webgpu binary backwards")
    }
}
