use super::*;
use rand::{distributions::Distribution, Rng};

pub trait Randomize {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D);
}

fn randomize<T: IsShapedArray, R: Rng, D: Distribution<f32>>(t: &mut T, rng: &mut R, dist: &D) {
    t.mut_data().map_inplace(|f| *f = dist.sample(rng));
}

impl<H> Randomize for Tensor0D<H> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}

impl<const N: usize, H> Randomize for Tensor1D<N, H> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}

impl<const M: usize, const N: usize, H> Randomize for Tensor2D<M, N, H> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}

impl<const M: usize, const N: usize, const O: usize, H> Randomize for Tensor3D<M, N, O, H> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, H> Randomize
    for Tensor4D<M, N, O, P, H>
{
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}
