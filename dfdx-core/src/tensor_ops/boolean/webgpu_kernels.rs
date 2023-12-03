use crate::prelude::Webgpu;

impl super::BooleanKernel for Webgpu {
    fn not<S: crate::prelude::Shape>(
        &self,
        inp: &crate::prelude::Tensor<S, bool, Self>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }

    fn and<S: crate::prelude::Shape>(
        &self,
        lhs: &crate::prelude::Tensor<S, bool, Self>,
        rhs: &crate::prelude::Tensor<S, bool, Self>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }

    fn or<S: crate::prelude::Shape>(
        &self,
        lhs: &crate::prelude::Tensor<S, bool, Self>,
        rhs: &crate::prelude::Tensor<S, bool, Self>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }

    fn xor<S: crate::prelude::Shape>(
        &self,
        lhs: &crate::prelude::Tensor<S, bool, Self>,
        rhs: &crate::prelude::Tensor<S, bool, Self>,
    ) -> Result<crate::prelude::Tensor<S, bool, Self>, crate::prelude::Error> {
        todo!()
    }
}
