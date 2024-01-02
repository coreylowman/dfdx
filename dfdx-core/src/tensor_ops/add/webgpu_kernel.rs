use std::borrow::Cow;

use crate::prelude::{
    ops::{BinaryKernel, UnaryKernel},
    Dtype, Webgpu,
};
use crate::tensor_ops::webgpu_kernels::wgpu_binary;
use super::BinaryAddKernelOp as Binary;

const BADD_SRC: &'static str = include_str!("./binary_add.wgsl");
wgpu_binary!(
    const_df() Binary,
    f32,
    BADD_SRC,
    "badd",
    "badd_fwd_f32",
    "badd_bwd_lhs_f32",
    "badd_bwd_rhs_f32"
);

// putting these tests here b/c I haven't added support for other data types
// and/or unary adds yet, so using the mod's tests breaks the build
#[cfg(test)]
mod tests {
    use crate::{shapes::*, tensor::*, tensor_ops::*, tests::*};
    #[test]
    fn test_add_zeroes() {
        let dev: Webgpu = Default::default();

        let a: Tensor<Rank3<2, 3, 4>, f32, _> = dev.zeros();
        let b: Tensor<Rank3<2, 3, 4>, f32, _> = dev.ones();
        let actual = a + b.clone();
        let actual_vec = actual.as_vec();
        let expected = b.as_vec();

        assert_eq!(actual_vec, expected);
    }

    #[test]
    fn test_add_increasing() {
        let dev: Webgpu = Default::default();

        let a = dev.tensor(&[1.0f32, 2.0, 3.0]);
        let result = a.clone() + a.clone();
        let result = a + result.clone();
        assert_eq!(result.as_vec(), &[3.0, 6.0, 9.0]);
    }
}
// impl<E: Dtype> UnaryKernel<super::ScalarAddKernelOp<E>, E> for Webgpu {
//     const BACKWARD_WITHOUT_INP: bool = false;

//     const BACKWARD_WITHOUT_DATA: bool = true;

//     fn forward<S: crate::prelude::Shape>(
//         &self,
//         op: super::ScalarAddKernelOp<E>,
//         inp: Cow<crate::prelude::Tensor<S, E, Self>>,
//     ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
//         todo!()
//     }

//     fn backward<S: crate::prelude::Shape>(
//         &self,
//         op: super::ScalarAddKernelOp<E>,
//         inp: &impl crate::prelude::Tensorlike<S, E, Self>,
//         grad_inp: &mut Self::Vec,
//         out: &impl crate::prelude::Tensorlike<S, E, Self>,
//         grad_out: &Self::Vec,
//     ) -> Result<(), crate::prelude::Error> {
//         todo!()
//     }
// }

// impl<E: Dtype> BinaryKernel<super::BinaryAddKernelOp, E> for Webgpu {
//     const BACKWARD_WITHOUT_DATA: bool = true;

//     fn forward<S: crate::prelude::Shape>(
//         &self,
//         op: super::BinaryAddKernelOp,
//         lhs: Cow<crate::prelude::Tensor<S, E, Self>>,
//         rhs: Cow<crate::prelude::Tensor<S, E, Self>>,
//     ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
//         todo!()
//     }

//     fn backward<S: crate::prelude::Shape>(
//         &self,
//         op: super::BinaryAddKernelOp,
//         lhs: &impl crate::prelude::Tensorlike<S, E, Self>,
//         grad_lhs: &mut Self::Vec,
//         rhs: &impl crate::prelude::Tensorlike<S, E, Self>,
//         grad_rhs: &mut Self::Vec,
//         grad_out: &Self::Vec,
//     ) -> Result<(), crate::prelude::Error> {
//         todo!()
//     }
// }
