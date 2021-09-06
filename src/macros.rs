#[macro_export]
macro_rules! call_with_all {
    ($owner:ident [$first:tt, ], $fn_name:ident($($fn_args:ident, )*)) => {
        $owner.$first.$fn_name($($fn_args, )*);
    };

    ($owner:ident [$first:tt, $($rest:tt, )*], $fn_name:ident($($fn_args:ident, )*)) => {
        $owner.$first.$fn_name($($fn_args, )*);
        call_with_all!($owner [$($rest, )*], $fn_name($($fn_args, )*));
    };
}

#[macro_export]
macro_rules! module_collection {
    ($typename:ident, [$($modules:tt, )*]) => {
        module_collection!([], [], $typename, [$($modules, )*], []);
    };

    ([$($const_defs:tt)*], [$($consts:tt)*], $typename:ident, [$($modules:tt, )*]) => {
        module_collection!([$($const_defs)*], [$($consts)*], $typename, [$($modules, )*], []);
    };

    ([$($const_defs:tt)*], [$($consts:tt)*], $typename:ident, [$($modules:tt, )*], [$($where_clauses:tt)*]) => {
        use $crate::call_with_all;
        use $crate::gradients::GradientTape;
        use $crate::traits::{RandomInit, Params};
        use ndarray_rand::rand::prelude::*;
        impl<$($const_defs)*> RandomInit for $typename<$($consts)*> where $($where_clauses)* {
            fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
                call_with_all!(self [$($modules, )*], randomize(rng, dist,));
            }
        }

        impl<$($const_defs)*> Params for $typename<$($consts)*> where $($where_clauses)* {
            fn register(&mut self, tape: &mut GradientTape) {
                call_with_all!(self [$($modules, )*], register(tape,));
            }

            fn update(&mut self, tape: &GradientTape) {
                call_with_all!(self [$($modules, )*], update(tape,));
            }
        }
    };
}
