use super::tape::GradientTape;

pub trait Taped {
    fn update(&mut self, tape: &GradientTape);
}
