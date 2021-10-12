#[derive(Debug, Clone, Copy)]
pub(crate) struct DerivativeRef {
    pub(super) index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GradientRef {
    pub(super) index: usize,
}
