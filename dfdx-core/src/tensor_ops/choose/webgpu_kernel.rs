use crate::prelude::{Dtype, Webgpu};

impl<E: Dtype> super::ChooseKernel<E> for Webgpu {
    fn forward<S: crate::prelude::Shape>(
        &self,
        cond: &crate::prelude::Tensor<S, bool, Self>,
        lhs: &crate::prelude::Tensor<S, E, Self>,
        rhs: &crate::prelude::Tensor<S, E, Self>,
    ) -> Result<crate::prelude::Tensor<S, E, Self>, crate::prelude::Error> {
        todo!()
    }

    fn backward<S: crate::prelude::Shape>(
        &self,
        cond: &crate::prelude::Tensor<S, bool, Self>,
        lhs: &crate::prelude::Tensor<S, E, Self>,
        grad_lhs: &mut <Self as crate::prelude::Storage<E>>::Vec,
        rhs: &crate::prelude::Tensor<S, E, Self>,
        grad_rhs: &mut <Self as crate::prelude::Storage<E>>::Vec,
        grad_out: &<Self as crate::prelude::Storage<E>>::Vec,
    ) -> Result<(), crate::prelude::Error> {
        todo!()
    }
}
