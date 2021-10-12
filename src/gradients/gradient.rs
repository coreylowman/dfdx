use super::refs::GradientRef;
use super::tape::GradientTape;

#[derive(Debug)]
pub struct Gradient {
    pub(crate) gradient_ref: GradientRef,
    pub(crate) tape: Option<Box<GradientTape>>,
}

impl Gradient {
    pub(crate) fn new(gradient_ref: GradientRef) -> Self {
        Self {
            gradient_ref,
            tape: None,
        }
    }

    pub(crate) fn with_tape(gradient_ref: GradientRef, tape: Box<GradientTape>) -> Self {
        Self {
            gradient_ref,
            tape: Some(tape),
        }
    }
}
