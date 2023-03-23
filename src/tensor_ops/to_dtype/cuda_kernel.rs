use crate::{
    shapes::{Shape, Unit},
    tensor::{cuda::Cuda, launch_cfg, unique_id, Tensor},
};
use cudarc::driver::{DeviceSlice, LaunchAsync};
use std::{sync::Arc, vec::Vec};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/to_dtype.ptx"));
const MODULE_NAME: &str = "to_dtype";

// Minimize the amount of code in the macro to speed up compilation
trait DtypeConversion {
    const FUNC_NAME: &'static str;
}

macro_rules! conversion {
    ($e1:ty, $e2:ty) => {
        impl DtypeConversion for ($e1, $e2) {
            const FUNC_NAME: &'static str = concat!(stringify!($e1), "_to_", stringify!($e2));
        }
    };
}

impl<E1: Unit, E2: Unit> super::ToDtypeKernel<E1, E2> for Cuda
where
    (E1, E2): DtypeConversion,
{
    fn forward<S: Shape>(inp: Tensor<S, E1, Self>) -> Result<Tensor<S, E2, Self>, Self::Err> {
        let fn_name = <(E1, E2)>::FUNC_NAME;
        let dev = &inp.device.dev;

        if !dev.has_func(MODULE_NAME, fn_name) {
            dev.load_ptx(PTX.into(), MODULE_NAME, &all_fn_names())?;
        }
        let numel = inp.data.len();
        let mut out = unsafe { dev.alloc::<E2>(numel) }?;

        let fwd_fn = dev.get_func(MODULE_NAME, fn_name).unwrap();
        let cfg = launch_cfg(numel as u32);
        let params = (numel, inp.data.as_ref(), &mut out);

        unsafe { fwd_fn.launch(cfg, params) }?;

        Ok(Tensor {
            id: unique_id(),
            data: Arc::new(out),
            shape: inp.shape,
            strides: inp.strides,
            device: inp.device.clone(),
            tape: Default::default(),
        })
    }
}

// Recursive macro that computes the names of all kernel functions and produces a function that
// outputs them
macro_rules! all_fn_names {
    // perform "nested for loop" with the elements of from_types and to_types to produce the final
    // list
    (@ $([$from_types:ident, [$($to_types:ident),+]])+) => {
        [$($(concat!(stringify!($from_types), "_to_", stringify!($to_types))),+),+]
    };
    // repeat to_types for each from_type
    (@ [$($from_types:ident),+] $to_types:tt) => {
        all_fn_names!(@ $([$from_types, $to_types])+)
    };
    ($from_types:tt $to_types:tt) => {
        fn all_fn_names() -> Vec<&'static str> {
            all_fn_names!(@ $from_types $to_types).to_vec()
        }
    }
}

// recursive macro to call "conversions" with all pairs of elements in from_types and to_types
macro_rules! all_conversions {
    (@ $from_type:ident [$($to_types:ident),*]) => {
        $(conversion!($from_type, $to_types);)*
    };
    ([$($from_types:ident),+] $to_types:tt) => {
        all_fn_names!([$($from_types),+] $to_types);
        $(all_conversions!(@ $from_types $to_types);)+
    }
}

all_conversions!(
    [f32, f64, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, bool]
    [f32, f64, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize]
);
