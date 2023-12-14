// TODO: We need to figure out how to represent empty structs in wgsl
// struct AbsKernelOp {
//     empty: u32,
// }

@group(0)
@binding(0)
var<storage, read> op: array<f32>;

@group(0)
@binding(1)
var<storage, read> inp: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> out: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> inp_grad: array<f32>;

@group(0)
@binding(4)
var<storage, read_write> out_grad: array<f32>;

@compute 
@workgroup_size(1)
fn abs_fwd_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // let length: u32 = arrayLength(&inp);
    // if (length > 1) {
    //     out[global_id.x] = abs(inp[global_id.x]);
    // } else {
    //     out[global_id.x] = abs(out[global_id.x]);
    // }
    out[global_id.x] = abs(inp[global_id.x]);
}

@compute
@workgroup_size(1)
fn abs_bwd_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Not needed for Abs, but if we can figure out a template system, we can leave it all in.
    // let x = if arrayLength(inp) > 0 { inp[global_id] } else { 0.0 };
    // let y = if arrayLength(out) > 0 { out[global_id] } else { 0.0 };
    var dx: f32;
    dx = sign(inp[global_id.x]);

    inp_grad[global_id.x] += dx * out_grad[global_id.x];
}
