/// Conversion trait for SafeTensors dtype
pub trait SafeTensorsDtype {
    #[cfg(feature = "safetensors")]
    const DTYPE: safetensors::tensor::Dtype;
}

impl<T: SafeTensorsDtype> SafeTensorsDtype for super::AMP<T> {
    #[cfg(feature = "safetensors")]
    const DTYPE: safetensors::tensor::Dtype = T::DTYPE;
}

macro_rules! dtype {
    ($type:ty, $dtype:expr) => {
        impl SafeTensorsDtype for $type {
            #[cfg(feature = "safetensors")]
            const DTYPE: safetensors::tensor::Dtype = $dtype;
        }
    };
}

dtype!(bool, safetensors::tensor::Dtype::BOOL);
dtype!(f32, safetensors::tensor::Dtype::F32);
dtype!(f64, safetensors::tensor::Dtype::F64);
dtype!(u8, safetensors::tensor::Dtype::U8);
dtype!(u16, safetensors::tensor::Dtype::U16);
dtype!(u32, safetensors::tensor::Dtype::U32);
dtype!(u64, safetensors::tensor::Dtype::U64);
dtype!(u128, safetensors::tensor::Dtype::U64);
dtype!(i8, safetensors::tensor::Dtype::I8);
dtype!(i16, safetensors::tensor::Dtype::I16);
dtype!(i32, safetensors::tensor::Dtype::I32);
dtype!(i64, safetensors::tensor::Dtype::I64);
dtype!(i128, safetensors::tensor::Dtype::I64);
#[cfg(feature = "f16")]
dtype!(super::f16, safetensors::tensor::Dtype::F16);

impl SafeTensorsDtype for usize {
    #[cfg(feature = "safetensors")]
    const DTYPE: safetensors::tensor::Dtype = safetensors::tensor::Dtype::U64;
}

impl SafeTensorsDtype for isize {
    #[cfg(feature = "safetensors")]
    const DTYPE: safetensors::tensor::Dtype = safetensors::tensor::Dtype::I64;
}
