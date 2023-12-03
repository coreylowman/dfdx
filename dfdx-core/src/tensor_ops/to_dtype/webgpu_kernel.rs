use crate::prelude::{Unit, Webgpu};

impl<E1: Unit, E2: Unit> super::ToDtypeKernel<E1, E2> for Webgpu {
    fn forward<S: crate::prelude::Shape>(
        inp: crate::prelude::Tensor<S, E1, Self>,
    ) -> Result<crate::prelude::Tensor<S, E2, Self>, crate::prelude::Error> {
        todo!()
    }
}
