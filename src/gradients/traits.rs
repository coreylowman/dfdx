use super::tape::GradientTape;

pub trait Params {
    fn update(&mut self, tape: &GradientTape);
}
