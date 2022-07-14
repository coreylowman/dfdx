use dfdx::prelude::*;
use indicatif::ProgressBar;
use mnist::*;
use rand::prelude::{SeedableRng, StdRng};
use std::time::Instant;

struct MnistDataset {
    img: Vec<f32>,
    lbl: Vec<usize>,
}

impl MnistDataset {
    fn train(path: &str) -> Self {
        let mnist: Mnist = MnistBuilder::new().base_path(path).finalize();
        Self {
            img: mnist.trn_img.iter().map(|&v| v as f32 / 255.0).collect(),
            lbl: mnist.trn_lbl.iter().map(|&v| v as usize).collect(),
        }
    }

    fn len(&self) -> usize {
        self.lbl.len()
    }

    pub fn get_batch<const B: usize>(
        &self,
        idxs: [usize; B],
    ) -> (Tensor2D<B, 784>, Tensor2D<B, 10>) {
        let mut img = Tensor2D::zeros();
        let mut lbl = Tensor2D::zeros();
        let img_data = img.mut_data();
        let lbl_data = lbl.mut_data();
        for (batch_i, &img_idx) in idxs.iter().enumerate() {
            let start = 784 * img_idx;
            img_data[batch_i].copy_from_slice(&self.img[start..start + 784]);
            lbl_data[batch_i][self.lbl[img_idx]] = 1.0;
        }
        (img, lbl)
    }
}


/// Custom model struct
/// This case is trivial and should be done with a tuple of linears and relus, 
/// but it demonstrates how to build models with custom behavior
#[derive(Default)]
struct Mlp<const IN: usize, const INNER: usize, const OUT: usize> {
    l1: Linear<IN, INNER>,
    l2: Linear<INNER, OUT>,
    relu: ReLU,
}

impl<const IN: usize, const INNER: usize, const OUT: usize>ResetParams for Mlp<IN, INNER, OUT> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.l1.reset_params(rng);
        self.l2.reset_params(rng);
        self.relu.reset_params(rng);
    }
}

impl<const IN: usize, const INNER: usize, const OUT: usize>CanUpdateWithGradients for Mlp<IN, INNER, OUT> {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.l1.update(grads);
        self.l2.update(grads);
        self.relu.update(grads);
    }
}

impl<const IN: usize, const INNER: usize, const OUT: usize> Module<Tensor1D<IN>> for Mlp<IN, INNER, OUT> {
    type Output = Tensor1D<OUT>;

    fn forward(&self, input: Tensor1D<IN>) -> Self::Output {
        self.l2.forward(
            self.relu.forward(
                self.l1.forward(input)
            )
        )
    }
}

impl<const BATCH: usize, const IN: usize, const INNER: usize, const OUT: usize> Module<Tensor2D<BATCH, IN, OwnedTape>> for Mlp<IN, INNER, OUT> {
    type Output = Tensor2D<BATCH, OUT, OwnedTape>;

    fn forward(&self, input: Tensor2D<BATCH, IN, OwnedTape>) -> Self::Output {
        self.l2.forward(
            self.relu.forward(
                self.l1.forward(input)
            )
        )
    }
}

const BATCH_SIZE: usize = 32;

fn main() {
    let mnist_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./datasets/MNIST/raw".to_string());

    println!("Loading mnist from args[1] = {mnist_path}");
    println!("Override mnist path with `cargo run --example mnist_classifier -- <path to mnist>`");

    let mut rng = StdRng::seed_from_u64(0);

    let mut model: Mlp<784, 512, 10> = Mlp::default();
    model.reset_params(&mut rng);
    let mut opt: Adam<Mlp<784, 512, 10>> = Default::default();

    let dataset = MnistDataset::train(&mnist_path);
    println!("Found {:?} training images", dataset.len());

    for i_epoch in 0..10 {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        let bar = ProgressBar::new(dataset.len() as u64);
        for (img, lbl) in SubsetIterator::<BATCH_SIZE>::shuffled(dataset.len(), &mut rng)
            .map(|i| dataset.get_batch(i))
        {
            let logits = model.forward(img.traced());
            let loss = cross_entropy_with_logits_loss(logits, &lbl);

            total_epoch_loss += loss.data();
            num_batches += 1;
            bar.inc(BATCH_SIZE as u64);

            opt.update(&mut model, loss.backward());
        }
        let dur = Instant::now() - start;
        bar.finish_and_clear();

        println!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.3}",
            dur,
            num_batches as f32 / dur.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
        );
    }
}