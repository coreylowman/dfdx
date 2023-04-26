//! This example is meant to show the following:
//! 1. How to house a model in a struct.
//! 2. How to seed a model for reproducibility.
//! 3. Getting the shape of a runtime dimensioned tensor.
//! 4. Dynamic batch size.

// ************************************************************************************************
// use
// ************************************************************************************************

use std::{cmp::min, time::Instant};

use dfdx::{
    data::{ExactSizeDataset, IteratorBatchExt, IteratorCollateExt, IteratorStackExt},
    nn::builders::*,
    optim::Adam,
    prelude::{mse_loss, Optimizer},
    shapes::{Const, HasShape, Rank1, Rank2},
    tensor::{AsArray, AutoDevice, Gradients, SampleTensor, Tensor, TensorFrom, Trace},
    tensor_ops::{Backward, BroadcastTo, RealizeTo, SelectTo, SumTo},
};
use indicatif::ProgressIterator;
use rand::{rngs::StdRng, Rng, SeedableRng};

// ************************************************************************************************
// Model definition
// ************************************************************************************************

type Mlp = (
    (Linear<2, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 1>, Sigmoid),
);

// ************************************************************************************************
// predictor definition
// ************************************************************************************************

pub struct Predictor {
    device: AutoDevice,
    model: <Mlp as BuildOnDevice<AutoDevice, f32>>::Built,
    gradients: Gradients<f32, AutoDevice>,
    optimizer: Adam<<Mlp as BuildOnDevice<AutoDevice, f32>>::Built, f32, AutoDevice>,
}

// ************************************************************************************************
// predictor API
// ************************************************************************************************

impl Predictor {
    pub fn new(seed: u64) -> Self {
        let device = AutoDevice::seed_from_u64(seed);
        let model = device.build_module::<Mlp, f32>();
        let gradients = model.alloc_grads();
        let optimizer: Adam<<Mlp as BuildOnDevice<AutoDevice, f32>>::Built, f32, AutoDevice> =
            Adam::new(&model, Default::default());
        Self {
            device,
            model,
            gradients,
            optimizer,
        }
    }

    pub fn predict_single(
        &self,
        input: Tensor<Rank1<2>, f32, AutoDevice>,
    ) -> Tensor<Rank1<1>, f32, AutoDevice> {
        // add one dimension to make the input look like a batch wit only one single entry.
        let batched: Tensor<Rank2<1, 2>, _, _> = input.clone().broadcast();

        // convert static size tensor to variable sized tensor
        let batched_realized: Tensor<(usize, Const<2>), _, _> = batched.realize().unwrap();
        assert_eq!(batched_realized.shape(), &(1 as usize, Const::<2>));

        // call predict on batches
        let batched_prediction = self.predict_batch(batched_realized);

        // batched_prediction should have the same number of batches as batched_realized
        assert_eq!(batched_prediction.shape(), &(1 as usize, Const::<1>));

        // get the result (batched_prediction[0])
        batched_prediction.select(self.device.tensor(0))
    }

    pub fn predict_batch(
        &self,
        input: Tensor<(usize, Const<2>), f32, AutoDevice>,
    ) -> Tensor<(usize, Const<1>), f32, AutoDevice> {
        self.model.forward(input)
    }

    // notice that we allow for batch size to change during training.
    pub fn learn_batch(
        &mut self,
        input: Tensor<(usize, Const<2>), f32, AutoDevice>,
        expected_output: Tensor<(usize, Const<1>), f32, AutoDevice>,
    ) -> f32 {
        assert_eq!(input.shape().0, expected_output.shape().0);

        let predictions = self
            .model
            .forward_mut(input.traced(self.gradients.to_owned()));
        let loss = mse_loss(predictions, expected_output);

        let batch_loss = loss.array();
        // num_batches += 1;

        self.gradients = loss.backward();
        self.optimizer
            .update(&mut self.model, &self.gradients)
            .unwrap();
        self.model.zero_grads(&mut self.gradients);
        batch_loss
    }
}

// ************************************************************************************************
// Building dataset
// ************************************************************************************************

fn function_we_would_like_the_nn_to_mimic(
    input: Tensor<(Const<2>,), f32, AutoDevice>,
) -> Tensor<(Const<1>,), f32, AutoDevice> {
    let distance_from_center: Tensor<(), f32, AutoDevice> = input.powi(2).sum().sqrt();
    if distance_from_center.as_vec()[0] > 1.0 {
        AutoDevice::default().tensor([1.0])
    } else {
        AutoDevice::default().tensor([0.0])
    }
}

struct XYPointsDataSet {
    points_and_predictions: Vec<(
        Tensor<Rank1<2>, f32, AutoDevice>,
        Tensor<Rank1<1>, f32, AutoDevice>,
    )>,
}

impl XYPointsDataSet {
    fn new(size: usize, seed: u64) -> Self {
        let device = AutoDevice::seed_from_u64(seed);
        let points_and_predictions = (0..size)
            .into_iter()
            .map(|_| {
                // create a random point in the (x, y) plane
                let point: Tensor<Rank1<2>, f32, AutoDevice> =
                    device.sample(rand_distr::Uniform::new(-2.0, 2.0));

                // get the classification f this point
                let class = function_we_would_like_the_nn_to_mimic(point.to_owned());

                // return point and classification
                (point, class)
            })
            .collect();
        Self {
            points_and_predictions,
        }
    }

    fn get_loss_of_predictor(&self, predictor: &Predictor) -> f32 {
        let mut total_epoch_loss = 0.0;
        for (points, classifications) in self
            .iter()
            // .map(preprocess)
            .batch_exact(self.len())
            .collate()
            .stack()
            .progress()
        {
            let predictions = predictor.predict_batch(points);
            let loss = mse_loss(predictions, classifications);
            total_epoch_loss += loss.array();
        }
        total_epoch_loss
    }
}

impl ExactSizeDataset for XYPointsDataSet {
    // define what an item in the dataset looks like
    type Item<'a> = (Tensor<Rank1<2>, f32, AutoDevice>, Tensor<Rank1<1>, f32, AutoDevice>) where Self: 'a;

    // get a specific item with an index in the dataset
    fn get(&self, index: usize) -> Self::Item<'_> {
        self.points_and_predictions[index].to_owned()
    }

    // get the length of the dataset
    fn len(&self) -> usize {
        self.points_and_predictions.len()
    }
}

// ************************************************************************************************
// using the predictor
// ************************************************************************************************

fn main() {
    let mut rng = StdRng::seed_from_u64(43);

    // create datasets
    let train_set = XYPointsDataSet::new(800, rng.gen());
    let test_set = XYPointsDataSet::new(200, rng.gen());
    let mut predictor = Predictor::new(rng.gen());

    for epoch_number in 1..100 {
        let mut num_batches = 0;
        let batch_size = min(epoch_number, 128);
        let mut total_epoch_loss = 0.0;

        let start = Instant::now();
        for (points, classifications) in train_set
            .shuffled(&mut rng)
            // .map(preprocess)
            .batch_exact(batch_size)
            .collate()
            .stack()
            .progress()
        {
            num_batches += 1;
            total_epoch_loss += predictor.learn_batch(points, classifications);
        }
        let duration = start.elapsed().as_secs_f32();

        println!(
            "Epoch {epoch_number} in {} seconds ({:.3} batches/s): avg sample loss {:.5}, test loss = {:.3}",
            duration,
            ((num_batches as f32) / duration),
            batch_size as f32 * total_epoch_loss / num_batches as f32,
            test_set.get_loss_of_predictor(&predictor)
        );
    }
}
