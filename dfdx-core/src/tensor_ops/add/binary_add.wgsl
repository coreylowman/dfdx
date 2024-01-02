alias usize = u32;

struct BinaryKernelMeta {
    numel: usize,
    num_dims: usize,
    info: array<usize>
}

@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

// @group(0) @binding(2)
// var<storage, read> kernel_meta: BinaryKernelMeta;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn badd_fwd_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    output[i] = lhs[i] + rhs[i];
}
