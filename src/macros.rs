#[macro_export]
macro_rules! call_with_all {
    ($owner:ident [$first:tt], $fn_name:ident($($fn_args:ident)*)) => {
        $owner.$first.$fn_name($($fn_args)*);
    };

    ($owner:ident [$first:tt $($rest:tt)*], $fn_name:ident($($fn_args:ident)*)) => {
        $owner.$first.$fn_name($($fn_args)*);
        call_with_all!($owner [$($rest)*], $fn_name($($fn_args)*));
    };
}

#[macro_export]
macro_rules! module_collection {
    ($typename:ident[$($modules:tt)*]) => {
        module_collection!([] [] $typename[$($modules)*]);
    };

    ([$($const_defs:tt)*] [$($consts:tt)*] $typename:ident[$($modules:tt)*]) => {
        use $crate::call_with_all;
        impl<$($const_defs)*> RandomInit for $typename<$($consts)*> {
            fn randomize<R: Rng>(&mut self, rng: &mut R) {
                call_with_all!(self [$($modules)*], randomize(rng));
            }
        }

        impl<$($const_defs)*> Params for $typename<$($consts)*> {
            fn register(&mut self, tape: &mut GradientTape) {
                call_with_all!(self [$($modules)*], register(tape));
            }

            fn update(&mut self, tape: &GradientTape) {
                call_with_all!(self [$($modules)*], update(tape));
            }
        }
    };
}
