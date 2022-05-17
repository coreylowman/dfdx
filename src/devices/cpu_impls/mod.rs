pub struct Cpu;

mod fill;
mod map;
mod map_inner;
mod reduce;
mod reduce_inner;
mod zero;
mod zip_map;

pub use fill::*;
pub use map::*;
pub use map_inner::*;
pub use reduce::*;
pub use reduce_inner::*;
pub use zero::*;
pub use zip_map::*;
