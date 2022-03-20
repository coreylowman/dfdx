use super::{structs::*, traits::*};
use rand::{distributions::Distribution, Rng};

fn randomize<T: IsShapedArray, R: Rng, D: Distribution<f32>>(t: &mut T, rng: &mut R, dist: &D) {
    t.mut_data().map_inplace(|f| *f = dist.sample(rng));
}

impl<Tape> Randomize for Tensor0D<Tape> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}

impl<const N: usize, Tape> Randomize for Tensor1D<N, Tape> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}

impl<const M: usize, const N: usize, Tape> Randomize for Tensor2D<M, N, Tape> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}

impl<const M: usize, const N: usize, const O: usize, Tape> Randomize for Tensor3D<M, N, O, Tape> {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}
impl<const M: usize, const N: usize, const O: usize, const P: usize, Tape> Randomize
    for Tensor4D<M, N, O, P, Tape>
{
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        randomize(self, rng, dist);
    }
}
