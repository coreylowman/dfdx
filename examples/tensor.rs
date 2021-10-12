use ndarray_rand::{
    rand::prelude::*,
    rand_distr::{Standard, StandardNormal},
};
use stag::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut x: Tensor2D<3, 3> = Default::default();
    println!("x={:#}", x.data());

    x.randomize(&mut rng, &Standard);
    println!("x={:#}", x.data());

    x.randomize(&mut rng, &StandardNormal);
    println!("x={:#}", x.data());

    println!("x.square().mean() = {:#}", x.square().mean().data());
    println!("x.relu()={:#}", x.relu().data());
    println!("x.tanh()={:#}", x.tanh().data());

    let mut y = Tensor2D::<3, 3>::randn(&mut rng);
    println!("y={:#}", y.data());
    println!("x+y={:#}", (&mut x + &mut y).data());

    let mut z = Tensor2D::<3, 2>::ones();
    println!("x@z={:#}", (&mut x * &mut z).data());
}
