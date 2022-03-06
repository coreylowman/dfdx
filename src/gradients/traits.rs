use super::structs::*;

pub trait OnGradientTape {
    fn update(&mut self, tape: &GradientTape);
}

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
