//! This example ties all the previous ones together
//! to build a neural network that learns to recognize
//! the MNIST digits.

use dfdx::data::SubsetIterator;
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

// our network structure
type Mlp = (
    (Linear<784, 512>, ReLU),
    (Linear<512, 128>, ReLU),
    (Linear<128, 32>, ReLU),
    Linear<32, 10>,
);

// training batch size
const BATCH_SIZE: usize = 32;

fn main() {
    // ftz substantially improves performance
    dfdx::flush_denormals_to_zero();

    let mnist_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./datasets/MNIST/raw".to_string());

    println!("Loading mnist from args[1] = {mnist_path}");
    println!("Override mnist path with `cargo run --example mnist_classifier -- <path to mnist>`");

    let mut rng = StdRng::seed_from_u64(0);

    // initialize model and optimizer
    let mut model: Mlp = Default::default();
    model.reset_params(&mut rng);
    let mut opt: Adam<Mlp> = Default::default();

    // initialize dataset
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
            let logits = model.forward_mut(img.traced());
            let loss = cross_entropy_with_logits_loss(logits, &lbl);

            total_epoch_loss += loss.data();
            num_batches += 1;
            bar.inc(BATCH_SIZE as u64);

            opt.update(&mut model, loss.backward())
                .expect("Unused params");
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

    // save our model to a .npz file
    model
        .save("mnist-classifier.npz")
        .expect("failed to save model");
}
