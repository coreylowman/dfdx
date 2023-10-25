use crate::shapes::ConstShape;

/// Marker for shapes that have the same number of elements as `Dst`
pub trait AssertSameNumel<Dst: ConstShape>: ConstShape {
    const TYPE_CHECK: ();
    fn assert_same_numel() {
        #[allow(clippy::let_unit_value)]
        let _ = <Self as AssertSameNumel<Dst>>::TYPE_CHECK;
    }
}

impl<Src: ConstShape, Dst: ConstShape> AssertSameNumel<Dst> for Src {
    const TYPE_CHECK: () = assert!(Src::NUMEL == Dst::NUMEL);
}
