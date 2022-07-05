use dfdx::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

type MultiHeadedMLP = (
    (Linear<10, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    SplitInto<((Linear<32, 2>, Tanh), (Linear<32, 1>, Tanh))>,
);

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    // initialize target data
    let x: Tensor2D<64, 10> = Tensor2D::randn(&mut rng);
    let y1: Tensor2D<64, 2> = Tensor2D::randn(&mut rng);
    let y2: Tensor2D<64, 1> = Tensor2D::randn(&mut rng);

    // initialize optimizer & model
    let mut mlp: MultiHeadedMLP = Default::default();
    mlp.reset_params(&mut rng);
    let mut sgd = Sgd::new(1e-2, None);

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        let x = x.trace();
        let (pred1, pred2) = mlp.forward(x);

        // NOTE: we also have to move the tape around when computing losses
        let (loss2, tape) = mse_loss(pred2, &y2).split_tape();
        let loss1 = mse_loss(pred1.put_tape(tape), &y1);

        let losses = [*loss1.data(), *loss2.data()];
        let loss = loss1 + &loss2;
        let gradients = loss.backward();
        sgd.update(&mut mlp, gradients);

        println!("losses={:.3?} in {:?}", losses, start.elapsed());
    }
}
