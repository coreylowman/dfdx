alias T = __SRC__;
alias U = __DST__;

@group(0) @binding(0)
var<storage, read> in: array<T>;

@group(0) @binding(1)
var<storage, read_write> out: array<U>;

@compute @workgroup_size(1, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    out[i] = U(in[i]);
}
