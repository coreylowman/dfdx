use super::Webgpu;
use wgpu;

const UNARY_OP_LAYOUT_NAME: &'static str = "unary";
const BINARY_OP_LAYOUT_NAME: &'static str = "binary";

impl Webgpu {
    #[inline]
    pub(crate) fn unary_op_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layouts[0]
    }

    #[inline]
    pub(crate) fn binary_op_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layouts[1]
    }

    /// Creates a [`wgpu::ComputePipeline`] for a binary operation.
    /// 
    /// todo: implement pipeline caching
    ///
    /// shader_name: the name of the shader module
    /// shader_source: the module's WGSL source code
    /// fn_name: The name of the entry point function
    pub(crate) fn load_binary_pipeline(
        &self,
        shader_name: &str,
        shader_source: &str,
        fn_name: &str,
    ) -> wgpu::ComputePipeline {
        self.load_pipeline(
            shader_name,
            shader_source,
            fn_name,
            &[self.binary_op_layout()],
        )
    }

    pub(crate) fn load_unary_pipeline(
        &self,
        shader_name: &str,
        shader_source: &str,
        fn_name: &str,
    ) -> wgpu::ComputePipeline {
        self.load_pipeline(
            shader_name,
            shader_source,
            fn_name,
            &[self.unary_op_layout()],
        )
    }

    /// Creates a [`wgpu::ComputePipeline`] for some operation.
    ///
    /// - `shader_name`: the name of the shader module
    /// - `shader_source`: the module's WGSL source code
    /// - `fn_name`: The name of the entry point function
    /// - `layouts`: bind group layouts
    pub(crate) fn load_pipeline(
        &self,
        shader_name: &str,
        shader_source: &str,
        fn_name: &str,
        layouts: &[&wgpu::BindGroupLayout],
    ) -> wgpu::ComputePipeline {
        // todo: cache these
        let source = wgpu::ShaderSource::Wgsl(shader_source.into());
        let shader_module = self.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_name),
            source,
        });

        let pipeline_layout = self
            .dev
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(fn_name),
                bind_group_layouts: layouts,
                // todo: these are useful and we should use them if the adapter supports them
                push_constant_ranges: &[],
            });

        self.dev
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(fn_name),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: fn_name,
            })
    }
}

pub(super) const fn unary_op_layout_desc() -> wgpu::BindGroupLayoutDescriptor<'static> {
    const ENTRIES: [wgpu::BindGroupLayoutEntry; 2] = [
        // input tensor buffer
        storage_entry(0, true),
        // TODO: metadata buffer (also try getting uniforms to work, since we can only have 8 storage buffers at once)
        // storage_entry(1, true),
        // output tensor buffer
        storage_entry(1, false),
    ];
    wgpu::BindGroupLayoutDescriptor {
        label: Some(UNARY_OP_LAYOUT_NAME),
        entries: &ENTRIES,
    }
}

pub(super) const fn binary_op_layout_desc() -> wgpu::BindGroupLayoutDescriptor<'static> {
    const ENTRIES: [wgpu::BindGroupLayoutEntry; 3] = [
        // lhs tensor buffer
        storage_entry(0, true),
        // rhs tensor buffer
        storage_entry(1, true),
        // TODO: metadata buffer (also try getting uniforms to work, since we can only have 8 storage buffers at once)
        // storage_entry(2, true),
        // output tensor buffer
        storage_entry(2, false),
    ];
    wgpu::BindGroupLayoutDescriptor {
        label: Some(BINARY_OP_LAYOUT_NAME),
        entries: &ENTRIES,
    }
}

/// Creates a [`wgpu::BindGroupLayoutEntry`] for a storage buffer. Useful for
/// composing a [`wgpu::BindGroupLayout`].
const fn storage_entry(index: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: index,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
