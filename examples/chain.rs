use ndarray_rand::rand::prelude::*;
use stag::nn::{Chain, Linear, ReLU};
use stag::prelude::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize a Linear layer followed by a ReLU activation
    // chain! expands to `<Linear<4, 2> as Default>::default().chain::<ReLU>()`;
    let mut model = chain!(Linear<4, 2>, ReLU);
    model.init(&mut rng);
    println!("{:?}", model);

    // create a 1x4 tensor with zeros
    let mut x: Tensor1D<4> = Tensor1D::default();

    // forward through the model
    let y = model.forward(&mut x);

    println!("{:#}", y.data());
    // [0.741256, 0]
}
