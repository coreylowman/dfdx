use crate::prelude::*;


pub struct BatchNorm1D<const C: usize> {
    pub gamma: Tensor1D<C, NoneTape>,
    pub beta: Tensor1D<C, NoneTape>,
    pub momentum: f32,
    pub epsilon: f32,
    pub running_mean: Tensor1D<C, NoneTape>,
    pub running_var: Tensor1D<C, NoneTape>,
}


impl<const C: usize> Default for BatchNorm1D<C> {
    /// Fills [Self::gamma] with 1s and [Self::beta] with 0s, sets [Self::momentum] to `0.1`,
    /// sets [Self::epsilon] to `1e-5`
    fn default() -> Self {
        Self {
            gamma: Tensor1D::ones(),
            beta: Tensor1D::zeros(),
            momentum: 0.1,
            epsilon: 1e-5,
            running_mean: Tensor1D::zeros(),
            running_var: Tensor1D::ones(),
        }
    }
}


impl<const C: usize> ResetParams for BatchNorm1D<C> {
    /// Fills [Self::gamma] with 1s and [Self::beta] with 0s.
    fn reset_params<R: rand::Rng>(&mut self, _: &mut R) {
        Cpu::fill(self.gamma.mut_data(), &mut |v| *v = 1.0);
        Cpu::fill(self.beta.mut_data(), &mut |v| *v = 0.0);
        Cpu::fill(self.running_mean.mut_data(), &mut |v| *v = 0.0);
        Cpu::fill(self.running_var.mut_data(), &mut |v| *v = 1.0);
    }
}


impl<const C: usize> CanUpdateWithGradients for BatchNorm1D<C> {
    /// Updates [Self::gamma] and [Self::beta].
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.gamma.update(grads, unused);
        self.beta.update(grads, unused);
    }
}


impl<const C: usize> Module<Tensor1D<C, NoneTape>> for BatchNorm1D<C> {
    type Output = Tensor1D<C, NoneTape>;

    /// Forward implementation for NoneTape
    /// Uses the tracked statistics to normalize the input
    /// Then scale using the gamma and beta parameters
    fn forward(&self, x: Tensor1D<C, NoneTape>) -> Self::Output {
        let sd = sqrt(add_scalar(self.running_var.duplicate(), self.epsilon));
        let x = div(sub(x.duplicate(), &self.running_mean.duplicate()), &sd);
        add(self.beta.duplicate(), &mul(self.gamma.duplicate(), &x))
    }
}


impl<const B: usize, const C: usize> Module<Tensor2D<B, C, NoneTape>> for BatchNorm1D<C> {
    type Output = Tensor2D<B, C, NoneTape>;

    /// Forward implementation for NoneTape
    /// Uses the tracked statistics to normalize the input
    /// Then scale using the broadcasted gamma and beta parameters
    fn forward(&self, x: Tensor2D<B, C, NoneTape>) -> Self::Output {
        let sd = sqrt(add_scalar(self.running_var.duplicate(), self.epsilon));
        let x = div(sub(x.duplicate(), &self.running_mean.duplicate().broadcast1()), &sd.broadcast1());
        add(self.beta.duplicate().broadcast1(), &mul(self.gamma.duplicate().broadcast1(), &x))
    }
}


impl<const B: usize, const C: usize, const L: usize> Module<Tensor3D<B, C, L, NoneTape>> for BatchNorm1D<C> {
    type Output = Tensor3D<B, C, L, NoneTape>;

    /// Forward implementation for NoneTape
    /// Uses the tracked statistics to normalize the input
    /// Then scale using the broadcasted gamma and beta parameters
    fn forward(&self, x: Tensor3D<B, C, L, NoneTape>) -> Self::Output {
        let sd = sqrt(add_scalar(self.running_var.duplicate(), self.epsilon));
        let x = div(sub(x.duplicate(), &self.running_mean.duplicate().broadcast2()), &sd.broadcast2());
        add(self.beta.duplicate().broadcast2(), &mul(self.gamma.duplicate().broadcast2(), &x))
    }
}


impl<const B: usize, const C: usize> Module<Tensor2D<B, C, OwnedTape>> for BatchNorm1D<C> {
    type Output = Tensor2D<B, C, OwnedTape>;

    // TODO Is this the right thing to do? The input has an OwnedTape
    // so statistcs needs to be tracked, but forward is being called
    // instead of forward_mut, should we not allow users to call forward here?
    // Should we use the running stats instead of batch statistics?
    /// Same as forward_mut but does not update statistics.
    fn forward(&self, x: Tensor2D<B, C, OwnedTape>) -> Self::Output {
        let (x, tape) = x.normalize_axis::<0>(self.epsilon).split_tape();
        let g: Tensor2D<B, C, OwnedTape> = self.gamma.duplicate().put_tape(tape).broadcast1();
        let (x, tape) = mul(g, &x).split_tape();
        let b = self.beta.duplicate().put_tape(tape).broadcast1();
        add(b, &x)
    }

    /// Normalize the input using batch mean and batch standard devation
    /// Update the running mean and running variance statistics
    /// Scale the normalized input using beta and gamma parameters
    fn forward_mut(&mut self, x: Tensor2D<B, C, OwnedTape>) -> Self::Output {
        let (x, tape) = x.normalize_axis::<0>(self.epsilon).split_tape();
        let batch_mean = x.duplicate().mean_axis::<0>();
        let batch_var = x.duplicate().var_axis::<0>();
        self.running_mean = add(batch_mean * self.momentum, &(self.running_mean.duplicate() * (1.0 - self.momentum)));
        self.running_var = add(batch_var * self.momentum, &(self.running_var.duplicate() * (1.0 - self.momentum)));
        let g: Tensor2D<B, C, OwnedTape> = self.gamma.duplicate().put_tape(tape).broadcast1();
        let (x, tape) = mul(g, &x).split_tape();
        let b = self.beta.duplicate().put_tape(tape).broadcast1();
        add(b, &x)
    }
}


impl<const B: usize, const C: usize, const L: usize> Module<Tensor3D<B, C, L, OwnedTape>> for BatchNorm1D<C> {
    type Output = Tensor3D<B, C, L, OwnedTape>;

    fn forward(&self, x: Tensor3D<B, C, L, OwnedTape>) -> Self::Output {
        let (x, tape) = x.split_tape();
        // TODO Implement reductions across multiple axis to make this cleaner/easier ?
        // We need mean across (B, L) slices
        let sample_mean = div_scalar(x.duplicate().sum_axis::<0>().sum_axis::<-1>(), (B + L) as f32);
        let (sample_var, tape) = div_scalar(
            square(sub(x.duplicate().put_tape(tape), &sample_mean.duplicate().broadcast2())).sum_axis::<0>().sum_axis::<-1>(),
            (B + L) as f32,
        ).split_tape();
        let sample_sd = sqrt(add_scalar(sample_var.duplicate(), self.epsilon));
        let (x, tape) = div(sub(x.duplicate().put_tape(tape), &sample_mean.duplicate().broadcast2()), &sample_sd.broadcast2()).split_tape();
        let g: Tensor3D<B, C, L, OwnedTape> = self.gamma.duplicate().put_tape(tape).broadcast2();
        let (x, tape) = mul(g, &x).split_tape();
        let b = self.beta.duplicate().put_tape(tape).broadcast2();
        add(b, &x)
    }

    fn forward_mut(&mut self, x: Tensor3D<B, C, L, OwnedTape>) -> Self::Output {
        let (x, tape) = x.split_tape();
        let sample_mean = div_scalar(x.duplicate().sum_axis::<0>().sum_axis::<-1>(), (B + L) as f32);
        let (sample_var, tape) = div_scalar(
            square(sub(x.duplicate().put_tape(tape), &sample_mean.duplicate().broadcast2())).sum_axis::<0>().sum_axis::<-1>(),
            (B + L) as f32,
        ).split_tape();
        self.running_mean = add(sample_mean.duplicate() * self.momentum, &(self.running_mean.duplicate() * (1.0 - self.momentum)));
        self.running_var = add(sample_var.duplicate() * self.momentum, &(self.running_var.duplicate() * (1.0 - self.momentum)));
        let sample_sd = sqrt(add_scalar(sample_var.duplicate(), self.epsilon));
        let (x, tape) = div(sub(x.duplicate().put_tape(tape), &sample_mean.duplicate().broadcast2()), &sample_sd.broadcast2()).split_tape();
        let g: Tensor3D<B, C, L, OwnedTape> = self.gamma.duplicate().put_tape(tape).broadcast2();
        let (x, tape) = mul(g, &x).split_tape();
        let b = self.beta.duplicate().put_tape(tape).broadcast2();
        add(b, &x)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::StdRng, SeedableRng};
    use rand_distr::Standard;

    #[test]
    fn test_batch_norm_reset() {
        let mut m: BatchNorm1D<5> = Default::default();
        assert_eq!(m.gamma.data(), &[1.0; 5]);
        assert_eq!(m.beta.data(), &[0.0; 5]);
        assert_eq!(m.running_mean.data(), &[0.0; 5]);
        assert_eq!(m.running_var.data(), &[1.0; 5]);

        let mut rng = StdRng::seed_from_u64(0);
        m.gamma.randomize(&mut rng, &Standard);
        m.beta.randomize(&mut rng, &Standard);

        assert_ne!(m.gamma.data(), &[1.0; 5]);
        assert_ne!(m.beta.data(), &[0.0; 5]);
        assert_eq!(m.running_mean.data(), &[0.0; 5]);
        assert_eq!(m.running_var.data(), &[1.0; 5]);

        m.reset_params(&mut rng);
        assert_eq!(m.gamma.data(), &[1.0; 5]);
        assert_eq!(m.beta.data(), &[0.0; 5]);
        assert_eq!(m.running_mean.data(), &[0.0; 5]);
        assert_eq!(m.running_var.data(), &[1.0; 5]);
    }
}
