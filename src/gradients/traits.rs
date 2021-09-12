use super::tape::GradientTape;

pub trait Params {
    fn register(&mut self, tape: &mut GradientTape);
    fn update(&mut self, tape: &GradientTape);
}
