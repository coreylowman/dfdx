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

use std::time::Instant;

use indicatif::ProgressIterator;
use mnist::*;
use rand::prelude::{SeedableRng, StdRng};

use dfdx::{data::*, optim::Adam, prelude::*, tensor::AutoDevice};

struct MnistTrainSet(Mnist);

impl MnistTrainSet {
    fn new(path: &str) -> Self {
        Self(MnistBuilder::new().base_path(path).finalize())
    }
}

impl ExactSizeDataset for MnistTrainSet {
    type Item<'a> = (Vec<f32>, usize) where Self: 'a;
    fn get(&self, index: usize) -> Self::Item<'_> {
        let mut img_data: Vec<f32> = Vec::with_capacity(784);
        let start = 784 * index;
        img_data.extend(
            self.0.trn_img[start..start + 784]
                .iter()
                .map(|x| *x as f32 / 255.0),
        );
        (img_data, self.0.trn_lbl[index] as usize)
    }
    fn len(&self) -> usize {
        self.0.trn_lbl.len()
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

    let dev = AutoDevice::default();
    let mut rng = StdRng::seed_from_u64(0);

    // initialize model, gradients, and optimizer
    let mut model = dev.build_module::<Mlp, f32>();
    let mut grads = model.alloc_grads();
    let mut opt = Adam::new(&model, Default::default());

    // initialize dataset
    let dataset = MnistTrainSet::new(&mnist_path);
    println!("Found {:?} training images", dataset.len());

    let preprocess = |(img, lbl): <MnistTrainSet as ExactSizeDataset>::Item<'_>| {
        let mut one_hotted = [0.0; 10];
        one_hotted[lbl] = 1.0;
        (
            dev.tensor_from_vec(img, (Const::<784>,)),
            dev.tensor(one_hotted),
        )
    };

    for i_epoch in 0..10 {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        for (img, lbl) in dataset
            .shuffled(&mut rng)
            .map(preprocess)
            .batch(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = model.forward_mut(img.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, lbl);

            total_epoch_loss += loss.array();
            num_batches += 1;

            grads = loss.backward();
            opt.update(&mut model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }
        let dur = Instant::now() - start;

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
