//! Intro to dfdx::gradients and tapes

use std::borrow::Borrow;
use rand::prelude::*;

use dfdx::gradients::{Gradients, NoneTape, OwnedTape};
use dfdx::tensor::{Tensor0D, Tensor2D, TensorCreator};
use dfdx::tensor_ops::matmul;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // tensors are first created with no tapes on them - the NoneTape!
    let weight: Tensor2D<4, 2, NoneTape> = TensorCreator::randn(&mut rng);
    let a: Tensor2D<3, 4, NoneTape> = TensorCreator::randn(&mut rng);

    // the first step to tracing is to call .trace()
    // this sticks a gradient tape into the input tensor!
    let b: Tensor2D<3, 4, OwnedTape> = a.trace();

    // the tape will automatically move around as you perform ops
    let c: Tensor2D<3, 2, OwnedTape> = matmul(b, &weight);
    let d: Tensor2D<3, 2, OwnedTape> = c.sin();
    let e: Tensor0D<OwnedTape> = d.mean();

    // finally you can use .backward() to extract the gradients!
    let gradients: Gradients = e.backward();

    // now you can extract gradients for specific tensors
    // by querying with them
    let weight_grad: &[[f32; 2]; 4] = gradients.ref_gradient(&weight);
    dbg!(weight_grad);

    let a_grad: &[[f32; 4]; 3] = gradients.ref_gradient(&a);
    dbg!(a_grad);

    let d_grad: &[[f32; 2]; 3] = gradients.ref_gradient(&d);
    println!("{:?}", d_grad);

    let e_grad: &f32 = gradients.ref_gradient(&e);
    println!("{:?}", e_grad);


}
