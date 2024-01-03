use super::AbsKernelOp;
use crate::tensor_ops::webgpu_kernels::webgpu_unary;

const F32_SPV_FWD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/abs.fwd.float.spv"));
const F32_SPV_BWD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/abs.bwd.float.spv"));
const F64_SPV_FWD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/abs.fwd.double.spv"));
const F64_SPV_BWD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/abs.bwd.double.spv"));

webgpu_unary!(AbsKernelOp, f32, F32_SPV_FWD, F32_SPV_BWD);
webgpu_unary!(AbsKernelOp, f64, F64_SPV_FWD, F64_SPV_BWD);

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_webgpu_abs() {
        let dev: Webgpu = Default::default();
        let x = dev.tensor([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let r = x.leaky_trace().abs();
        assert_close_to_literal!(r, [2.0, 1.0, 0.0, 1.0, 2.0]);
        // TODO: Add mean back in
        // let g = r.mean().backward();
        // assert_close_to_literal!(g.get(&x), [-0.2, -0.2, 0.0, 0.2, 0.2]);
    }
}
