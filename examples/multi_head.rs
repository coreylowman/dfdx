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

impl CanUpdateWithTape for MultiHeadedMLP {
    fn update_with_tape(&mut self, tape: &GradientTape) {
        self.trunk.update_with_tape(tape);
        self.head1.update_with_tape(tape);
        self.head2.update_with_tape(tape);
    }
}

impl<H: TapeHolder> Module<Tensor1D<10, H>> for MultiHeadedMLP {
    type Output = (Tensor1D<2, H>, Tensor1D<1, NoTape>);
    fn forward(&self, x: Tensor1D<10, H>) -> Self::Output {
        let x = self.trunk.forward(x);
        let (x, _x) = x.duplicate();
        let out2 = self.head2.forward(x);
        let (out2, tape_holder) = out2.split_tape_holder();
        let x = _x.with_tape_holder(tape_holder);
        let out1 = self.head1.forward(x);
        (out1, out2)
    }
}

impl<H: TapeHolder, const B: usize> Module<Tensor2D<B, 10, H>> for MultiHeadedMLP {
    type Output = (Tensor2D<B, 2, H>, Tensor2D<B, 1, NoTape>);
    fn forward(&self, x: Tensor2D<B, 10, H>) -> Self::Output {
        let x = self.trunk.forward(x);
        let (x, _x) = x.duplicate();
        let out2 = self.head2.forward(x);
        let (out2, tape_holder) = out2.split_tape_holder();
        let x = _x.with_tape_holder(tape_holder);
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
    let mut module: MultiHeadedMLP = Default::default();
    module.randomize(&mut rng, &Uniform::new(-1.0, 1.0));

    let mut sgd = Sgd { lr: 1e-2 };

    // run through training data
    for _i_epoch in 0..15 {
        let start = Instant::now();

        let x = x.with_tape();
        let (pred1, pred2) = module.forward(x);
        let (loss1, tape_holder) = mse_loss(pred1, &y1).split_tape_holder();
        let loss2 = mse_loss(pred2.with_tape_holder(tape_holder), &y2);
        let losses = [loss1.data()[()], loss2.data()[()]];
        let loss = &loss1 + loss2;
        let (loss_v, gradients) = sgd.compute_gradients(loss);
        module.update_with_tape(&gradients);

        println!(
            "mse={:#.3} (losses={:.3?}) in {:?}",
            loss_v,
            losses,
            start.elapsed()
        );
    }
}
