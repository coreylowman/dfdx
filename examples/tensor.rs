use rand::prelude::*;
use rand_distr::{Standard, StandardNormal};
use stag::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // empty 3x3 matrix
    let mut x: Tensor2D<3, 3> = Default::default();
    println!("x={:#}", x.data());

    // fill matrix with random data drawn from Standard distribution
    x.randomize(&mut rng, &Standard);
    println!("x={:#}", x.data());

    // fill matrix with random data drawn from StandardNormal distribution
    x.randomize(&mut rng, &StandardNormal);
    println!("x={:#}", x.data());

    // call some functions on the matrix
    println!("x.square().mean() = {:#}", x.square().mean().data());
    println!("x.relu()={:#}", x.relu().data());
    println!("x.tanh()={:#}", x.tanh().data());

    // add two tensors of same size together
    let mut y: Tensor2D<3, 3> = Tensor2D::randn(&mut rng);
    println!("y={:#}", y.data());
    println!("x+y={:#}", add(&mut x, &mut y).data());

    // multiply two tensors with same inner dimension
    let mut z: Tensor2D<3, 2> = Tensor2D::ones();
    println!("x@z={:#}", matmat_mul(&mut x, &mut z).data());
}
