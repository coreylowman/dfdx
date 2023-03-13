use super::*;

/// Marker for shapes that can be converted using their concrete types.
pub trait RealizeShapeTo<Dst: Shape>: Shape {
    fn realized(&self) -> Option<Dst>;
}

impl<Src: Shape<Concrete = Dst::Concrete>, Dst: Shape> RealizeShapeTo<Dst> for Src {
    #[inline(always)]
    fn realized(&self) -> Option<Dst> {
        Dst::from_concrete(&self.concrete())
    }
}
