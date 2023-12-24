use super::AbsKernelOp;
use crate::tensor_ops::webgpu_kernels::webgpu_unary;

const GLSL_FWD: &str = include_str!("abs.fwd.glsl");
const GLSL_BWD: &str = include_str!("abs.bwd.glsl");
const SPV_FWD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/abs.fwd.float.spv"));
const SPV_BWD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/abs.bwd.float.spv"));

webgpu_unary!(AbsKernelOp, f32, SPV_FWD, SPV_BWD);

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_webgpu_abs() {
        let dev: Webgpu = Default::default();
        let x = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().abs();
        assert_close_to_literal!(r, [2.0, 1.0, 0.0, 1.0, 2.0]);
        // TODO: Add mean back in
        // let g = r.mean().backward();
        // assert_close_to_literal!(g.get(&x), [-0.2, -0.2, 0.0, 0.2, 0.2]);
    }
}
