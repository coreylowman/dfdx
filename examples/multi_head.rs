use dfdx::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

#[derive(Default)]
struct MultiHeadedMLP {
    trunk: ((Linear<10, 32>, ReLU), (Linear<32, 32>, ReLU)),
    head1: (Linear<32, 2>, Tanh),
    head2: (Linear<32, 1>, Tanh),
}

impl ResetParams for MultiHeadedMLP {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.trunk.reset_params(rng);
        self.head1.reset_params(rng);
        self.head2.reset_params(rng);
    }
}

impl CanUpdateWithGradients for MultiHeadedMLP {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.trunk.update(grads);
        self.head1.update(grads);
        self.head2.update(grads);
    }
}

impl<H: Tape, const B: usize> Module<Tensor2D<B, 10, H>> for MultiHeadedMLP {
    type Output = (Tensor2D<B, 2, H>, Tensor2D<B, 1, NoTape>);
    fn forward(&self, x: Tensor2D<B, 10, H>) -> Self::Output {
        // execute trunk
        let x = self.trunk.forward(x);

        // duplicate x to use later for the other head
        let _x = x.duplicate();

        // pass x through the 2nd head
        let out2 = self.head2.forward(x);

        // grab the tape out of out2, so we can put it back into our duplicate of x
        let (out2, tape) = out2.split_tape();

        // put the tape holder back into the duplicate of x
        let x = _x.put_tape(tape);

        // now pass the tape through the other head
        let out1 = self.head1.forward(x);

        (out1, out2)
    }
}

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
        let (loss1, tape) = mse_loss(pred1, &y1).split_tape();
        let loss2 = mse_loss(pred2.put_tape(tape), &y2);

        let losses = [*loss1.data(), *loss2.data()];
        let loss = &loss1 + loss2;
        let gradients = loss.backward();
        sgd.update(&mut mlp, gradients);

        println!("losses={:.3?} in {:?}", losses, start.elapsed());
    }
}
