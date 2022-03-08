use super::structs::*;

pub trait OnGradientTape {
    fn put_on(&mut self, tape: &mut GradientTape);
    fn update_with(&mut self, tape: &GradientTape);
}

pub trait HasGradientTape {
    fn tape(&self) -> &Option<Box<GradientTape>>;
    fn mut_tape(&mut self) -> &mut Option<Box<GradientTape>>;
}

pub trait HasGradientRef {
    fn grad_ref(&self) -> &Option<GradientRef>;
    fn mut_grad_ref(&mut self) -> &mut Option<GradientRef>;
}
