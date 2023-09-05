#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

mod layers;
pub mod optim;

pub use dfdx_nn_core::*;
pub use dfdx_nn_derives::*;
pub use layers::*;

pub use dfdx;

#[cfg(test)]
pub(crate) mod tests {
    pub use num_traits::{Float, FromPrimitive, NumCast, Zero};

    #[cfg(not(feature = "cuda"))]
    pub type TestDevice = dfdx::tensor::Cpu;

    #[cfg(feature = "cuda")]
    pub type TestDevice = dfdx::tensor::Cuda;

    #[cfg(all(feature = "test-f64", feature = "test-f16"))]
    compile_error!("f64 and f16 cannot be tested at the same time");

    #[cfg(all(
        not(feature = "test-amp-f16"),
        not(feature = "test-f16"),
        not(feature = "test-f64")
    ))]
    pub type TestDtype = f32;

    #[cfg(feature = "test-f16")]
    pub type TestDtype = dfdx::dtypes::f16;

    #[cfg(feature = "test-f64")]
    pub type TestDtype = f64;

    #[cfg(feature = "test-amp-f16")]
    pub type TestDtype = dfdx::dtypes::AMP<dfdx::dtypes::f16>;

    pub trait AssertClose {
        type Elem: std::fmt::Display + std::fmt::Debug + Copy;
        const DEFAULT_TOLERANCE: Self::Elem;
        fn get_default_tol(&self) -> Self::Elem {
            Self::DEFAULT_TOLERANCE
        }
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)>;
        fn assert_close(&self, rhs: &Self, tolerance: Self::Elem)
        where
            Self: std::fmt::Debug,
        {
            if let Some((l, r)) = self.get_far_pair(rhs, tolerance) {
                panic!("lhs != rhs | {l} != {r}\n\n{self:?}\n\n{rhs:?}");
            }
        }
    }

    impl<F: Copy + std::fmt::Debug + std::fmt::Display + AssertClose> AssertClose
        for dfdx::dtypes::AMP<F>
    {
        type Elem = dfdx::dtypes::AMP<F::Elem>;
        const DEFAULT_TOLERANCE: Self::Elem = dfdx::dtypes::AMP(F::DEFAULT_TOLERANCE);
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)> {
            self.0
                .get_far_pair(&rhs.0, tolerance.0)
                .map(|(l, r)| (dfdx::dtypes::AMP(l), dfdx::dtypes::AMP(r)))
        }
    }

    #[cfg(feature = "f16")]
    impl AssertClose for half::f16 {
        type Elem = Self;
        const DEFAULT_TOLERANCE: Self::Elem = half::f16::from_f32_const(1e-2);
        fn get_far_pair(&self, rhs: &Self, tolerance: Self) -> Option<(Self, Self)> {
            if num_traits::Float::abs(self - rhs) > tolerance {
                Some((*self, *rhs))
            } else {
                None
            }
        }
    }

    impl AssertClose for f32 {
        type Elem = f32;
        const DEFAULT_TOLERANCE: Self::Elem = 1e-6;
        fn get_far_pair(&self, rhs: &Self, tolerance: f32) -> Option<(f32, f32)> {
            if (self - rhs).abs() > tolerance {
                Some((*self, *rhs))
            } else {
                None
            }
        }
    }

    impl AssertClose for f64 {
        type Elem = f64;
        const DEFAULT_TOLERANCE: Self::Elem = 1e-6;
        fn get_far_pair(&self, rhs: &Self, tolerance: f64) -> Option<(f64, f64)> {
            if (self - rhs).abs() > tolerance {
                Some((*self, *rhs))
            } else {
                None
            }
        }
    }

    impl<T: AssertClose, const M: usize> AssertClose for [T; M] {
        type Elem = T::Elem;
        const DEFAULT_TOLERANCE: Self::Elem = T::DEFAULT_TOLERANCE;
        fn get_far_pair(
            &self,
            rhs: &Self,
            tolerance: Self::Elem,
        ) -> Option<(Self::Elem, Self::Elem)> {
            for (l, r) in self.iter().zip(rhs.iter()) {
                if let Some(pair) = l.get_far_pair(r, tolerance) {
                    return Some(pair);
                }
            }
            None
        }
    }

    pub trait NdMap {
        type Elem;
        type Mapped<O>;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, f: F) -> Self::Mapped<O>;
    }

    impl NdMap for f32 {
        type Elem = Self;
        type Mapped<O> = O;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, mut f: F) -> O {
            f(self)
        }
    }

    impl NdMap for f64 {
        type Elem = Self;
        type Mapped<O> = O;
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, mut f: F) -> O {
            f(self)
        }
    }

    impl<T: NdMap, const M: usize> NdMap for [T; M] {
        type Elem = T::Elem;
        type Mapped<O> = [T::Mapped<O>; M];
        fn ndmap<O, F: Copy + FnMut(Self::Elem) -> O>(self, f: F) -> Self::Mapped<O> {
            self.map(|t| t.ndmap(f))
        }
    }

    macro_rules! assert_close_to_literal {
        ($Lhs:expr, $Rhs:expr) => {{
            let lhs = $Lhs.array();
            let rhs = $Rhs.ndmap(|x| num_traits::FromPrimitive::from_f64(x).unwrap());
            let tol = AssertClose::get_default_tol(&lhs);
            let far_pair = AssertClose::get_far_pair(&lhs, &rhs, tol);
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        }};
        ($Lhs:expr, $Rhs:expr, $Tolerance:expr) => {{
            let far_pair = $Lhs.array().get_far_pair(
                &$Rhs.ndmap(|x| num_traits::FromPrimitive::from_f64(x).unwrap()),
                num_traits::FromPrimitive::from_f64($Tolerance).unwrap(),
            );
            if let Some((l, r)) = far_pair {
                panic!("lhs != rhs | {l} != {r}");
            }
        }};
    }
    pub(crate) use assert_close_to_literal;
}
