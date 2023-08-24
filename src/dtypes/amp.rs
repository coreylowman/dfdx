use rand::{distributions::Distribution, Rng};

/// Wrapper type around the storage type. Use like `AMP<f16>` or `AMP<bf16>`.
///
/// This causes some tensor operations to cast the type to a higher precision
/// and then back. For example calling sum on a `AMP<f16>` tensor will cast it to
/// `f32`, sum it, and then cast it back to `f16`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct AMP<F>(pub F);

#[cfg(feature = "f16")]
impl AMP<half::f16> {
    pub const INFINITY: Self = AMP(half::f16::INFINITY);
    pub const NEG_INFINITY: Self = AMP(half::f16::NEG_INFINITY);
}

#[cfg(feature = "std")]
impl<F: std::fmt::Display> std::fmt::Display for AMP<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl<F: super::SafeZeros> super::SafeZeros for AMP<F> {}

#[cfg(feature = "cuda")]
unsafe impl<F: cudarc::driver::ValidAsZeroBits> cudarc::driver::ValidAsZeroBits for AMP<F> {}

#[cfg(feature = "cuda")]
unsafe impl<F: cudarc::driver::DeviceRepr> cudarc::driver::DeviceRepr for AMP<F> {}

#[cfg(feature = "cuda")]
impl<F: cudarc::types::CudaTypeName> cudarc::types::CudaTypeName for AMP<F> {
    const NAME: &'static str = F::NAME;
}

#[cfg(feature = "cudnn")]
impl<F: cudarc::cudnn::CudnnDataType> cudarc::cudnn::CudnnDataType for AMP<F> {
    type Scalar = F::Scalar;
    const DATA_TYPE: cudarc::cudnn::sys::cudnnDataType_t = F::DATA_TYPE;
    fn into_scaling_parameter(self) -> Self::Scalar {
        self.0.into_scaling_parameter()
    }
}

impl<F: std::ops::Add<F, Output = F>> std::ops::Add<AMP<F>> for AMP<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        AMP(self.0 + rhs.0)
    }
}

impl<F: std::ops::Sub<F, Output = F>> std::ops::Sub<AMP<F>> for AMP<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        AMP(self.0 - rhs.0)
    }
}

impl<F: std::ops::Mul<F, Output = F>> std::ops::Mul<AMP<F>> for AMP<F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        AMP(self.0 * rhs.0)
    }
}

impl<F: std::ops::Div<F, Output = F>> std::ops::Div<AMP<F>> for AMP<F> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        AMP(self.0 / rhs.0)
    }
}

impl<F: std::ops::Rem<F, Output = F>> std::ops::Rem<AMP<F>> for AMP<F> {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        AMP(self.0 % rhs.0)
    }
}

impl<F: std::ops::Neg<Output = F>> std::ops::Neg for AMP<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        AMP(-self.0)
    }
}

impl<'l, 'r, F> std::ops::Add<&'r AMP<F>> for &'l AMP<F>
where
    &'l F: std::ops::Add<&'r F, Output = F>,
{
    type Output = AMP<F>;
    fn add(self, rhs: &'r AMP<F>) -> Self::Output {
        AMP(&self.0 + &rhs.0)
    }
}

impl<'l, 'r, F> std::ops::Sub<&'r AMP<F>> for &'l AMP<F>
where
    &'l F: std::ops::Sub<&'r F, Output = F>,
{
    type Output = AMP<F>;
    fn sub(self, rhs: &'r AMP<F>) -> Self::Output {
        AMP(&self.0 - &rhs.0)
    }
}

impl<'l, 'r, F> std::ops::Mul<&'r AMP<F>> for &'l AMP<F>
where
    &'l F: std::ops::Mul<&'r F, Output = F>,
{
    type Output = AMP<F>;
    fn mul(self, rhs: &'r AMP<F>) -> Self::Output {
        AMP(&self.0 * &rhs.0)
    }
}

impl<'l, 'r, F> std::ops::Div<&'r AMP<F>> for &'l AMP<F>
where
    &'l F: std::ops::Div<&'r F, Output = F>,
{
    type Output = AMP<F>;
    fn div(self, rhs: &'r AMP<F>) -> Self::Output {
        AMP(&self.0 / &rhs.0)
    }
}

impl<'l, 'r, F> std::ops::Rem<&'r AMP<F>> for &'l AMP<F>
where
    &'l F: std::ops::Rem<&'r F, Output = F>,
{
    type Output = AMP<F>;
    fn rem(self, rhs: &'r AMP<F>) -> Self::Output {
        AMP(&self.0 % &rhs.0)
    }
}

impl<'l, F> std::ops::Neg for &'l AMP<F>
where
    &'l F: std::ops::Neg<Output = F>,
{
    type Output = AMP<F>;
    fn neg(self) -> Self::Output {
        AMP(-&self.0)
    }
}

impl<F: std::ops::AddAssign<F>> std::ops::AddAssign<AMP<F>> for AMP<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<F: std::ops::SubAssign<F>> std::ops::SubAssign<AMP<F>> for AMP<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<F: std::ops::MulAssign<F>> std::ops::MulAssign<AMP<F>> for AMP<F> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<F: std::ops::DivAssign<F>> std::ops::DivAssign<AMP<F>> for AMP<F> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<F: num_traits::FromPrimitive> num_traits::FromPrimitive for AMP<F> {
    fn from_f32(n: f32) -> Option<Self> {
        F::from_f32(n).map(AMP)
    }
    fn from_f64(n: f64) -> Option<Self> {
        F::from_f64(n).map(AMP)
    }
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(AMP)
    }
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(AMP)
    }
}

impl<T: 'static + Copy, F: num_traits::AsPrimitive<T>> num_traits::AsPrimitive<T> for AMP<F> {
    fn as_(self) -> T {
        self.0.as_()
    }
}

#[cfg(feature = "f16")]
impl num_traits::AsPrimitive<AMP<half::f16>> for half::f16 {
    fn as_(self) -> AMP<half::f16> {
        AMP(self)
    }
}

#[cfg(feature = "f16")]
impl num_traits::AsPrimitive<AMP<half::f16>> for f32 {
    fn as_(self) -> AMP<half::f16> {
        AMP(half::f16::from_f32(self))
    }
}

#[cfg(feature = "f16")]
impl num_traits::AsPrimitive<AMP<half::f16>> for f64 {
    fn as_(self) -> AMP<half::f16> {
        AMP(half::f16::from_f64(self))
    }
}

impl<F: num_traits::ToPrimitive> num_traits::ToPrimitive for AMP<F> {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
    fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }
    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

impl<F: crate::shapes::Unit> crate::shapes::Unit for AMP<F> {
    const ONE: Self = AMP(F::ONE);
}

impl<F: crate::shapes::Dtype> crate::shapes::Dtype for AMP<F> {}

impl<F: num_traits::Zero> num_traits::Zero for AMP<F> {
    fn zero() -> Self {
        AMP(F::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl<F: num_traits::One> num_traits::One for AMP<F> {
    fn one() -> Self {
        AMP(F::one())
    }
}
impl<F: num_traits::Num> num_traits::Num for AMP<F> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(AMP)
    }
}
impl<F: num_traits::NumCast> num_traits::NumCast for AMP<F> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(AMP)
    }
}
impl<F: num_traits::FloatConst> num_traits::FloatConst for AMP<F> {
    fn E() -> Self {
        AMP(F::E())
    }

    fn FRAC_1_PI() -> Self {
        AMP(F::FRAC_1_PI())
    }

    fn FRAC_1_SQRT_2() -> Self {
        AMP(F::FRAC_1_SQRT_2())
    }

    fn FRAC_2_PI() -> Self {
        AMP(F::FRAC_2_PI())
    }

    fn FRAC_2_SQRT_PI() -> Self {
        AMP(F::FRAC_2_SQRT_PI())
    }

    fn FRAC_PI_2() -> Self {
        AMP(F::FRAC_PI_2())
    }

    fn FRAC_PI_3() -> Self {
        AMP(F::FRAC_PI_3())
    }

    fn FRAC_PI_4() -> Self {
        AMP(F::FRAC_PI_4())
    }

    fn FRAC_PI_6() -> Self {
        AMP(F::FRAC_PI_6())
    }

    fn FRAC_PI_8() -> Self {
        AMP(F::FRAC_PI_8())
    }

    fn LN_10() -> Self {
        AMP(F::LN_10())
    }

    fn LN_2() -> Self {
        AMP(F::LN_2())
    }

    fn LOG10_E() -> Self {
        AMP(F::LOG10_E())
    }

    fn LOG2_E() -> Self {
        AMP(F::LOG2_E())
    }

    fn PI() -> Self {
        AMP(F::PI())
    }

    fn SQRT_2() -> Self {
        AMP(F::SQRT_2())
    }
}
impl<F: num_traits::Float> num_traits::Float for AMP<F> {
    fn nan() -> Self {
        AMP(F::nan())
    }

    fn infinity() -> Self {
        AMP(F::infinity())
    }

    fn neg_infinity() -> Self {
        AMP(F::neg_infinity())
    }

    fn neg_zero() -> Self {
        AMP(F::neg_zero())
    }

    fn min_value() -> Self {
        AMP(F::min_value())
    }

    fn min_positive_value() -> Self {
        AMP(F::min_positive_value())
    }

    fn max_value() -> Self {
        AMP(F::max_value())
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn classify(self) -> core::num::FpCategory {
        self.0.classify()
    }

    fn floor(self) -> Self {
        AMP(self.0.floor())
    }

    fn ceil(self) -> Self {
        AMP(self.0.ceil())
    }

    fn round(self) -> Self {
        AMP(self.0.round())
    }

    fn trunc(self) -> Self {
        AMP(self.0.trunc())
    }

    fn fract(self) -> Self {
        AMP(self.0.fract())
    }

    fn abs(self) -> Self {
        AMP(self.0.abs())
    }

    fn signum(self) -> Self {
        AMP(self.0.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        AMP(self.0.mul_add(a.0, b.0))
    }

    fn recip(self) -> Self {
        AMP(self.0.recip())
    }

    fn powi(self, n: i32) -> Self {
        AMP(self.0.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        AMP(self.0.powf(n.0))
    }

    fn sqrt(self) -> Self {
        AMP(self.0.sqrt())
    }

    fn exp(self) -> Self {
        AMP(self.0.exp())
    }

    fn exp2(self) -> Self {
        AMP(self.0.exp2())
    }

    fn ln(self) -> Self {
        AMP(self.0.ln())
    }

    fn log(self, base: Self) -> Self {
        AMP(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        AMP(self.0.log2())
    }

    fn log10(self) -> Self {
        AMP(self.0.log10())
    }

    fn max(self, other: Self) -> Self {
        AMP(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        AMP(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        AMP(self.0.abs_sub(other.0))
    }

    fn cbrt(self) -> Self {
        AMP(self.0.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        AMP(self.0.hypot(other.0))
    }

    fn sin(self) -> Self {
        AMP(self.0.sin())
    }

    fn cos(self) -> Self {
        AMP(self.0.cos())
    }

    fn tan(self) -> Self {
        AMP(self.0.tan())
    }

    fn asin(self) -> Self {
        AMP(self.0.asin())
    }

    fn acos(self) -> Self {
        AMP(self.0.acos())
    }

    fn atan(self) -> Self {
        AMP(self.0.atan())
    }

    fn atan2(self, other: Self) -> Self {
        AMP(self.0.atan2(other.0))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = self.0.sin_cos();
        (AMP(a), AMP(b))
    }

    fn exp_m1(self) -> Self {
        AMP(self.0.exp_m1())
    }

    fn ln_1p(self) -> Self {
        AMP(self.0.ln_1p())
    }

    fn sinh(self) -> Self {
        AMP(self.0.sinh())
    }

    fn cosh(self) -> Self {
        AMP(self.0.cosh())
    }

    fn tanh(self) -> Self {
        AMP(self.0.tanh())
    }

    fn asinh(self) -> Self {
        AMP(self.0.asinh())
    }

    fn acosh(self) -> Self {
        AMP(self.0.acosh())
    }

    fn atanh(self) -> Self {
        AMP(self.0.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }
}

macro_rules! impl_distribution {
    ($Distr:ty) => {
        impl<F> Distribution<AMP<F>> for $Distr
        where
            Self: Distribution<F>,
        {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> AMP<F> {
                AMP(<Self as Distribution<F>>::sample(self, rng))
            }
        }
    };
}

impl_distribution!(rand_distr::Standard);
impl_distribution!(rand_distr::StandardNormal);
impl_distribution!(rand_distr::Exp1);
impl_distribution!(rand_distr::Open01);
impl_distribution!(rand_distr::OpenClosed01);

#[derive(Debug, Clone, Copy)]
pub struct AMPSampler<F: rand_distr::uniform::SampleUniform>(F::Sampler);

impl<F: rand_distr::uniform::SampleUniform> rand_distr::uniform::SampleUniform for AMP<F> {
    type Sampler = AMPSampler<F>;
}

impl<F: rand_distr::uniform::SampleUniform> rand_distr::uniform::UniformSampler for AMPSampler<F> {
    type X = AMP<F>;
    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
    {
        let l = low.borrow();
        let h = high.borrow();
        Self(F::Sampler::new(&l.0, &h.0))
    }
    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand_distr::uniform::SampleBorrow<Self::X> + Sized,
    {
        let l = low.borrow();
        let h = high.borrow();
        Self(F::Sampler::new_inclusive(&l.0, &h.0))
    }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        AMP(self.0.sample(rng))
    }
}
