//! The struct definitions for all TensorXD, [Tensor] trait, and more.
//!
//! At a high level a tensor consists of only three parts
//! 1. A [crate::unique_id::UniqueId] to track which gradients are associated with what tensors
//! 2. An Nd rust array stored in a [std::sync::Arc].
//! 3. A tape, which can either actually be a tape ([crate::gradients::OwnedTape]) or be empty ([crate::gradients::NoneTape]).
//!
//! # Creating tensors
//!
//! ### Use the tensor function
//!
//! See [tensor()].
//!
//! ```rust
//! # use dfdx::prelude::*;
//! let t = tensor([1.0, 2.0, 3.0]);
//! ```
//!
//! ### Use the TensorCreator trait
//!
//! See [TensorCreator].
//!
//! 1. With raw rust arrays use the [TensorCreator::new()] method.
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor1D<3> = TensorCreator::new([1.0, 2.0, 3.0]);
//! ```
//!
//! 2. Filled with 0s or 1s use [TensorCreator::zeros()] and [TensorCreator::ones()].
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor1D<5> = TensorCreator::zeros();
//! let q: Tensor2D<3, 2> = TensorCreator::ones();
//! ```
//!
//! 3. Filled with random data use [TensorCreator::rand()] and [TensorCreator::randn()].
//! ```rust
//! # use dfdx::prelude::*;
//! # use rand::prelude::*;
//! let mut rng = StdRng::seed_from_u64(0);
//! let a = Tensor1D::<3>::rand(&mut rng); // uniform random data
//! let b = Tensor2D::<4, 3>::randn(&mut rng); // gaussian random data
//! ```
//!
//! # Accessing or modifying underlying data
//!
//! Use [crate::arrays::HasArrayData::data()] and [crate::arrays::HasArrayData::mut_data()]
//! to view or modify the underlying arrays.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! let mut t = Tensor1D::<3>::zeros();
//! assert_eq!(t.data(), &[0.0, 0.0, 0.0]);
//! t.mut_data()[1] = 2.0;
//! assert_eq!(t.data(), &[0.0, 2.0, 0.0]);
//! ```
//!
//! # Tracking gradients
//!
//! Use the [trace()] or [traced()] methods to add [crate::gradients::OwnedTape] to the [Tensor].
//! `.trace()` will clone the [crate::unique_id::UniqueId] & data, while `.traced()` will take ownership of
//! the tensor and return a version with an [crate::gradients::OwnedTape].
//!
//! Note that these two methods are only present for tensors without a tape already.
//!
//! ```rust
//! # use dfdx::prelude::*;
//! let t: Tensor1D<5, NoneTape> = Tensor1D::<5>::zeros();
//! let t_clone: Tensor1D<5, OwnedTape> = t.trace(); // copies t
//! let t: Tensor1D<5, OwnedTape> = t.traced(); // takes ownership of t
//! ```

mod base;
mod impl_alloc;
mod impl_update_with_grads;

pub use base::{Tensor, Tensor0D, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D, Tensor6D};
pub(crate) use impl_alloc::make_tensor;
pub use impl_alloc::TensorSugar;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{AsArray, AsVec, Ones, Rand, Randn, TryConvert, Zeros};
    use crate::gradients::{NoneTape, OwnedTape};
    use crate::tests::build_test_device;
    use crate::unique_id::{unique_id, UniqueId};
    use std::collections::HashSet;

    #[test]
    fn test_id() {
        let dev = build_test_device!();

        let mut ids: HashSet<UniqueId> = Default::default();
        ids.insert(unique_id());

        let x: Tensor0D<_> = dev.zeros();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);

        let x: Tensor0D<_> = dev.zeros();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);

        let x: Tensor1D<5, _> = dev.zeros();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);

        let x: Tensor2D<3, 2, _> = dev.ones();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);

        let x: Tensor3D<4, 3, 2, _> = dev.rand();
        assert!(!ids.contains(&x.id));
        ids.insert(x.id);
    }

    #[test]
    fn test_ids_with_clone() {
        let dev = build_test_device!();
        let t1: Tensor1D<32, _> = dev.zeros();
        let t2: Tensor1D<32, _> = t1.clone();
        assert_eq!(t1.id, t2.id);
    }

    #[test]
    fn test_ids_with_split_and_put() {
        let dev = build_test_device!();
        let t1: Tensor1D<32, _> = dev.zeros();
        let t1_id = t1.id;
        let (t2, tape) = t1.split_tape();
        assert_eq!(t2.id, t1_id);
        let t3 = t2.put_tape(tape);
        assert_eq!(t3.id, t1_id);
    }

    #[test]
    fn test_zeros() {
        let dev = build_test_device!();
        let x: Tensor2D<3, 2, _> = dev.zeros();
        assert_eq!(x.as_array(), [[0.0; 2]; 3]);
    }

    #[test]
    fn test_ones() {
        let dev = build_test_device!();
        let x: Tensor2D<3, 2, _> = dev.ones();
        assert_eq!(x.as_array(), [[1.0; 2]; 3]);
    }

    #[test]
    fn test_convert_array() {
        let dev = build_test_device!();
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let t: Tensor2D<2, 3, _> = dev.convert(a);
        assert_eq!(t.as_array(), a);
    }

    #[test]
    fn test_convert_slice() {
        let dev = build_test_device!();
        let data = [1.0, 2.0, 3.0, 4.0];
        let t: Tensor2D<2, 2, _> = dev.convert(data.as_slice());
        assert_eq!(t.as_array(), [[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn test_convert_vec() {
        let dev = build_test_device!();
        let data = std::vec![1.0, 2.0, 3.0, 4.0];
        let t: Tensor2D<2, 2, _> = dev.convert(data);
        assert_eq!(t.as_array(), [[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn fuzz_test_rand() {
        let dev = build_test_device!();
        let t: Tensor1D<1000, _> = dev.rand();
        for v in t.as_vec() {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_randn() {
        let dev = build_test_device!();
        let _: Tensor1D<1000, _> = dev.randn();
    }

    #[test]
    fn test_split_and_put() {
        let dev = build_test_device!();
        let a: Tensor0D<_, NoneTape> = dev.zeros();
        let b: Tensor0D<_, OwnedTape<_>> = a.traced();
        let (c, tape): (Tensor0D<_>, OwnedTape<_>) = b.split_tape();
        let d: Tensor0D<_, OwnedTape<_>> = c.put_tape(tape);
        let _: Tensor0D<_, OwnedTape<_>> = d.with_empty_tape();
        let _: Tensor0D<_, NoneTape> = d.with_diff_tape();
    }
}
