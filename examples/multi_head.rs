use rand::{distributions::Distribution, rngs::StdRng, Rng, SeedableRng};
use rand_distr::Uniform;
use stag::prelude::*;
use std::time::Instant;

#[derive(Default)]
struct MultiHeadedMLP {
    trunk: ((Linear<10, 32>, ReLU), (Linear<32, 32>, ReLU)),
    head1: (Linear<32, 2>, Tanh),
    head2: (Linear<32, 1>, Tanh),
}

impl Randomize for MultiHeadedMLP {
    fn randomize<R: Rng, D: Distribution<f32>>(&mut self, rng: &mut R, dist: &D) {
        self.trunk.randomize(rng, dist);
        self.head1.randomize(rng, dist);
        self.head2.randomize(rng, dist);
    }
}

impl CanUpdateWithGradients for MultiHeadedMLP {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.trunk.update(grads);
        self.head1.update(grads);
        self.head2.update(grads);
    }
}

impl<H: TapeHolder, const B: usize> Module<Tensor2D<B, 10, H>> for MultiHeadedMLP {
    type Output = (Tensor2D<B, 2, H>, Tensor2D<B, 1, NoTape>);
    fn forward(&self, x: Tensor2D<B, 10, H>) -> Self::Output {
        // execute trunk
        let x = self.trunk.forward(x);

        // duplicate x to use later for the other head
        let _x = x.duplicate();

        // pass x through the 2nd head
        let out2 = self.head2.forward(x);

        // grab the tape out of out2, so we can put it back into our duplicate of x
        let (out2, tape_holder) = out2.split_tape_holder();

        // put the tape holder back into the duplicate of x
        let x = _x.with_tape_holder(tape_holder);

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
    mlp.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    let mut sgd = Sgd::new(1e-2);

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        let x = x.trace();
        let (pred1, pred2) = mlp.forward(x);

        // NOTE: we also have to move the tape around when computing losses
        let (loss1, tape_holder) = mse_loss(pred1, &y1).split_tape_holder();
        let loss2 = mse_loss(pred2.with_tape_holder(tape_holder), &y2);

        let losses = [*loss1.data(), *loss2.data()];
        let loss = &loss1 + loss2;
        let gradients = loss.backward();
        sgd.update(&mut mlp, gradients);

        println!("losses={:.3?} in {:?}", losses, start.elapsed());
    }
}
