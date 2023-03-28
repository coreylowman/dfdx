// TODO

use cudarc::driver::{LaunchAsync, LaunchConfig};
use num_traits::Float;

use crate::prelude::{Cuda, Dtype, Shape, Tensor, Unit};

use super::PReLUKernel;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/prelu.ptx"));

trait HasCudaKernel<E: Unit> {
    const FN: &'static str;
    const FNB: &'static str;
}

impl HasCudaKernel<f32> for Cuda {
    const FN: &'static str = "prelu_fwd_f32";
    const FNB: &'static str = "prelu_bwd_f32";
}

// impl HasCudaKernel<f64> for Cuda {
//     const FN: &'static str = "prelu_fwd_f64";
//     const FNB: &'static str = "prelu_bwd_f64"
// }

impl<S: Shape, E: Dtype + Float> PReLUKernel<Tensor<S, E, Cuda>, Tensor<(), E, Cuda>> for Cuda
where
    Self: HasCudaKernel<E>,
{
    type Output = Tensor<S, E, Cuda>;

    type Elem = E;

    fn forward(
        &self,
        lhs: &Tensor<S, E, Cuda>,
        rhs: &Tensor<(), E, Cuda>,
    ) -> Result<Self::Output, <Self::Output as crate::prelude::HasErr>::Err> {
        if !self.dev.has_func(Self::FN, Self::FN) {
            self.dev.load_ptx(PTX.into(), Self::FN, &[Self::FN])?;
        }
        let f = self.dev.get_func(Self::FN, Self::FN).unwrap();

        let numel = lhs.shape.num_elements();
        let mut out_storage = self.dev.alloc_zeros::<E>(numel)?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,
            lhs.data.as_ref(),
            rhs.data.as_ref(),
            &mut out_storage,
        );

        unsafe { f.launch(cfg, params) }?;
        let out = self.build_tensor(lhs.shape, lhs.strides, out_storage);
        Ok(out)
    }

    fn backward(
        &self,
        lhs: &Tensor<S, E, Cuda>,
        lhs_grad: &mut <Self as crate::prelude::storage_traits::DeviceStorage>::Vec<Self::Elem>,
        rhs: &Tensor<(), E, Cuda>,
        rhs_grad: &mut <Self as crate::prelude::storage_traits::DeviceStorage>::Vec<Self::Elem>,
        grad: &<Self as crate::prelude::storage_traits::DeviceStorage>::Vec<Self::Elem>,
    ) -> Result<(), <Self::Output as crate::prelude::HasErr>::Err> {
        if !self.dev.has_func(Self::FNB, Self::FNB) {
            self.dev.load_ptx(PTX.into(), Self::FNB, &[Self::FNB])?;
        }
        let f = self.dev.get_func(Self::FNB, Self::FNB).unwrap();

        let numel = lhs.shape.num_elements();

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,
            lhs.data.as_ref(),
            lhs_grad,
            rhs.data.as_ref(),
            rhs_grad,
            grad,
        );

        unsafe { f.launch(cfg, params) }?;
        Ok(())
    }
}
