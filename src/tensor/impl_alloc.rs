use super::base::Tensor;
use crate::arrays::{Dtype, Shape};
use crate::devices::{
    AsArray, AsVec, Device, Ones, OnesLike, Rand, RandLike, Randn, RandnLike, TryConvert, Zeros,
    ZerosLike,
};
use crate::gradients::NoneTape;
use crate::unique_id::unique_id;

impl<S: Shape, E: Dtype, D: Device> Zeros<Tensor<S, E, D, NoneTape>> for D
where
    Self: Zeros<Self::Storage<S, E>>,
{
    fn try_zeros(&self) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_zeros()?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
    fn fill_with_zeros(&self, t: &mut Tensor<S, E, D, NoneTape>) {
        self.fill_with_zeros(&mut t.storage);
    }
}

impl<S: Shape, E: Dtype, D: Device> ZerosLike<S, Tensor<S, E, D, NoneTape>> for D
where
    Self: ZerosLike<S, Self::Storage<S, E>>,
{
    fn try_zeros_like(&self, src: S) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_zeros_like(src)?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
}

impl<S: Shape, E: Dtype, D: Device> Ones<Tensor<S, E, D, NoneTape>> for D
where
    Self: Ones<Self::Storage<S, E>>,
{
    fn try_ones(&self) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_ones()?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
    fn fill_with_ones(&self, t: &mut Tensor<S, E, D, NoneTape>) {
        self.fill_with_ones(&mut t.storage);
    }
}

impl<S: Shape, E: Dtype, D: Device> OnesLike<S, Tensor<S, E, D, NoneTape>> for D
where
    Self: OnesLike<S, Self::Storage<S, E>>,
{
    fn try_ones_like(&self, src: S) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_ones_like(src)?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
}

impl<S: Shape, E: Dtype, D: Device> Rand<Tensor<S, E, D, NoneTape>> for D
where
    Self: Rand<Self::Storage<S, E>>,
{
    fn try_rand(&self) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_rand()?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
    fn fill_with_rand(&self, t: &mut Tensor<S, E, D, NoneTape>) {
        self.fill_with_rand(&mut t.storage);
    }
}

impl<S: Shape, E: Dtype, D: Device> RandLike<S, Tensor<S, E, D, NoneTape>> for D
where
    Self: RandLike<S, Self::Storage<S, E>>,
{
    fn try_rand_like(&self, src: S) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_rand_like(src)?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
}

impl<S: Shape, E: Dtype, D: Device> Randn<Tensor<S, E, D, NoneTape>> for D
where
    Self: Randn<Self::Storage<S, E>>,
{
    fn try_randn(&self) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_randn()?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
    fn fill_with_randn(&self, t: &mut Tensor<S, E, D, NoneTape>) {
        self.fill_with_randn(&mut t.storage);
    }
}

impl<S: Shape, E: Dtype, D: Device> RandnLike<S, Tensor<S, E, D, NoneTape>> for D
where
    Self: RandnLike<S, Self::Storage<S, E>>,
{
    fn try_randn_like(&self, src: S) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_randn_like(src)?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
}

impl<S: Shape, Src, E: Dtype, D: Device> TryConvert<Src, Tensor<S, E, D, NoneTape>> for D
where
    Self: TryConvert<Src, Self::Storage<S, E>>,
{
    fn try_from(&self, src: Src) -> Result<Tensor<S, E, D, NoneTape>, Self::Err> {
        Ok(Tensor {
            id: unique_id(),
            storage: self.try_from(src)?,
            device: self.clone(),
            tape: NoneTape,
        })
    }
}

impl<S: Shape, E: Dtype, D: Device, T> AsVec for Tensor<S, E, D, T>
where
    D::Storage<S, E>: AsVec,
{
    type Vec = <D::Storage<S, E> as AsVec>::Vec;
    fn as_vec(&self) -> Self::Vec {
        self.storage.as_vec()
    }
}

impl<S: Shape, E: Dtype, D: Device, T> AsArray for Tensor<S, E, D, T>
where
    D::Storage<S, E>: AsArray,
{
    type Array = <D::Storage<S, E> as AsArray>::Array;
    fn as_array(&self) -> Self::Array {
        self.storage.as_array()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::unique_id::UniqueId;
//     use rand::thread_rng;
//     use std::collections::HashSet;

//     #[test]
//     fn test_id() {
//         let mut ids: HashSet<UniqueId> = Default::default();
//         ids.insert(unique_id());

//         let x = Tensor0D::new(0.0);
//         assert!(!ids.contains(&x.id));
//         ids.insert(x.id);

//         let x = Tensor0D::new(0.0);
//         assert!(!ids.contains(&x.id));
//         ids.insert(x.id);

//         let x = Tensor1D::<5>::zeros();
//         assert!(!ids.contains(&x.id));
//         ids.insert(x.id);

//         let x = Tensor2D::<3, 2>::ones();
//         assert!(!ids.contains(&x.id));
//         ids.insert(x.id);

//         let x = Tensor3D::<4, 3, 2>::zeros();
//         assert!(!ids.contains(&x.id));
//         ids.insert(x.id);
//     }

//     #[test]
//     fn test_zeros() {
//         assert_eq!(Tensor2D::<3, 2>::zeros().data(), &[[0.0; 2]; 3]);
//     }

//     #[test]
//     fn test_ones() {
//         assert_eq!(Tensor2D::<3, 2>::ones().data(), &[[1.0; 2]; 3]);
//     }

//     #[test]
//     fn test_new() {
//         let t = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//         assert_eq!(Tensor2D::new(t).data(), &t);
//     }

//     #[test]
//     fn fuzz_test_rand() {
//         let mut rng = thread_rng();
//         for &v in Tensor1D::<1000>::rand(&mut rng).data() {
//             assert!((0.0..1.0).contains(&v));
//         }
//     }

//     #[test]
//     fn test_randn() {
//         let mut rng = thread_rng();
//         let _t = Tensor1D::<1000>::randn(&mut rng);
//     }
// }
