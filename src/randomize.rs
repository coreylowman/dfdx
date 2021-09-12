use ndarray_rand::rand::{distributions::Distribution, Rng};

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}
