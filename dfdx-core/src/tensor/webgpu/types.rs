use crate::shapes::Unit;

/// A primitive data type natively supported by WebGPU.
///
/// See: https://www.w3.org/TR/WGSL/#types
///
/// todo: support packed types
pub trait WebgpuNativeType: Unit {
    /// Name of the data type in WGSL.
    const NAME: &'static str;
}

macro_rules! webgpu_type {
    ($RustTy:ty) => {
        impl WebgpuNativeType for $RustTy {
            const NAME: &'static str = stringify!($RustTy);
        }
    };
    ($RustTy:ty, $WgpuTy:expr) => {
        impl WebgpuNativeType for $RustTy {
            const NAME: &'static str = $WgpuTy;
        }
    };
}

/*
see:
- https://docs.rs/wgpu/latest/wgpu/struct.Features.html#associatedconstant.SHADER_F16
- https://docs.rs/wgpu/latest/wgpu/struct.Features.html#associatedconstant.SHADER_F64
- https://docs.rs/wgpu/latest/wgpu/struct.Features.html#associatedconstant.SHADER_I16
 */
#[cfg(feature = "f16")]
webgpu_type!(half::f16, "f16");
webgpu_type!(f32);
// todo: only enable when f64 feature is enabled
#[cfg(feature = "f64")]
webgpu_type!(f64);

#[cfg(feature = "i16")]
webgpu_type!(i16);
webgpu_type!(i32);

webgpu_type!(u32);
webgpu_type!(bool);

pub(crate) trait HasGlslType {
    const TYPE: &'static str;
}

impl HasGlslType for f32 {
    const TYPE: &'static str = "float";
}

impl HasGlslType for f64 {
    const TYPE: &'static str = "double";
}
