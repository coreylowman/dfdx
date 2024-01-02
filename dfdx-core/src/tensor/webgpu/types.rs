use crate::shapes::Unit;

/// A primitive data type natively supported by WebGPU.
pub trait WebgpuNativeType : Unit {
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

// pub trait WgpuPackedType<E: WgpuNativeType> : Unit {
//     const NAME: &'static str;
//     /// Number of elements packed into a single `E` value.
//     /// 
//     /// For example, `i8` is packed 4 times into a single `i32` value.
//     const PACK_WIDTH: usize;
// }
