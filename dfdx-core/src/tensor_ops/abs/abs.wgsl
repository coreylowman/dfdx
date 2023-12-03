struct AbsKernelOp {};

@group(0)
@binding(0)
var<storage, read> op: AbsKernelOp;

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
    var<private> x = if arrayLength(inp) > 0 { &inp[global_id] } else { &out[global_id] };
    *x = abs(*x);
}

@compute
@workgroup_size(1)
fn abs_bwd_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Not needed for Abs, but if we can figure out a template system, we can leave it all in.
    // let x = if arrayLength(inp) > 0 { inp[global_id] } else { 0.0 };
    // let y = if arrayLength(out) > 0 { out[global_id] } else { 0.0 };
    var<private> dx: f32;
    dx = sign(inp[global_id]);

    inp_grad[global_id] += dx * out_grad[global_id];
}
