use super::refs::GradientRef;
use super::tape::GradientTape;

pub trait HasGradient {
    fn grad(&self) -> &Option<Gradient>;
    fn mut_grad(&mut self) -> &mut Option<Gradient>;

    fn gradient_ref(&self) -> GradientRef {
        self.grad().as_ref().unwrap().gradient_ref
    }

    fn take_tape(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad()
            .as_mut()
            .map(|grad| grad.tape.take())
            .flatten()
    }

    fn backward(&mut self) -> Option<Box<GradientTape>> {
        self.mut_grad().as_mut().map(|grad| {
            let mut tape = grad.tape.take().unwrap();
            tape.backward(grad.gradient_ref);
            tape
        })
    }
}

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
