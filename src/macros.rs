#[macro_export]
macro_rules! chain {
    ($first:ty, $($rest:ty),+) => {
        {
            <$first as Default>::default()
            $(
                .chain::<$rest>()
            )*
        }
    };
}
