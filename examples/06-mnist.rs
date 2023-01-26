//! This example ties all the previous ones together
//! to build a neural network that learns to recognize
//! the MNIST digits.
//!
//! To download the MNIST dataset, do the following:
//! ```
//! mkdir tmp/ && cd tmp/
//! curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
//!     -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
//!     -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
//!     -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
//! gunzip t*-ubyte.gz
//! cd -
//! ```
//! Then, you may run this example with:
//! ```
//! cargo run --example 06-mnist -- tmp/
//! ```

use dfdx::{data::SubsetIterator, losses::cross_entropy_with_logits_loss, optim::Adam, prelude::*};
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
        dev: &AutoDevice,
        idxs: [usize; B],
    ) -> (
        Tensor<Rank2<B, 784>>,
        Tensor<Rank2<B, 10>>,
    ) {
        let mut img_data: Vec<f32> = Vec::with_capacity(B * 784);
        let mut lbl_data: Vec<f32> = Vec::with_capacity(B * 10);
        for (_batch_i, &img_idx) in idxs.iter().enumerate() {
            let start = 784 * img_idx;
            img_data.extend(&self.img[start..start + 784]);
            let mut choices = [0.0; 10];
            choices[self.lbl[img_idx]] = 1.0;
            lbl_data.extend(choices);
        }
        let mut img = dev.zeros();
        img.copy_from(&img_data);
        let mut lbl = dev.zeros();
        lbl.copy_from(&lbl_data);
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
    println!("Override mnist path with `cargo run --example 06-mnist -- <path to mnist>`");

    let dev: AutoDevice = Default::default();
    let mut rng = StdRng::seed_from_u64(0);

    // initialize model and optimizer
    let mut model: Mlp = dev.build_module();
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
            .map(|i| dataset.get_batch(&dev, i))
        {
            let logits = model.forward_mut(img.traced());
            let loss = cross_entropy_with_logits_loss(logits, lbl);

            total_epoch_loss += loss.array();
            num_batches += 1;
            bar.inc(BATCH_SIZE as u64);

            let gradients = loss.backward();
            opt.update(&mut model, gradients).unwrap();
        }
        let dur = Instant::now() - start;
        bar.finish_and_clear();

        println!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}",
            dur,
            num_batches as f32 / dur.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
        );
    }

    // save our model to a .npz file
    #[cfg(feature = "numpy")]
    model.save("06-mnist.npz").expect("failed to save model");
}
