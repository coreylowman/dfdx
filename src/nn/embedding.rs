use crate::{optim::*, shapes::*, tensor::*, tensor_ops::*};

use super::module::{Module, ModuleMut, ResetParams};

/// An embedding
/// Initializes [Self::weight] from a Uniform distribution
/// between [-1 / sqrt(I), 1 / sqrt(I)].
///
/// # Generics
/// - `VOCAB` The size of the vocabulary, inputs integer values must be between
///    0 and VOCAB;
/// - `DIM` The "output" size of vectors & matrices which are the vectors being selected.
///
/// # Examples
/// `Embedding<5, 2>` can act on vectors with SEQ integer elements (with values between 0 and 4), and results in a SEQ tensor of
/// usually f32 elements being the rows in [Self::weight].
/// ```rust
///
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let model: Embedding<5, 2> = dev.build_module();
/// // single item forward
/// let inputs: Tensor<(Const<5>,), usize> = dev.zeros::<Rank1<5>>();
/// let _: Tensor<(Const<2>,), f32> = model.forward(inputs);
/// // batched forward
/// let inputs: Tensor<(Const<10>, Const<5>,), usize> = dev.zeros::<Rank2<10, 5>>();
/// let _: Tensor<(Const<10>, Const<2>), f32> = model.forward(inputs);
/// ```
#[derive(Debug, Clone)]
pub struct Embedding<const VOCAB: usize, const DIM: usize, D: Device<f32> = Cpu> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<VOCAB, DIM>, f32, D>,
}

impl<const VOCAB: usize, const DIM: usize, const SEQ: usize, D: Device<f32>>
    Module<Tensor<Rank1<SEQ>, usize, D>> for Embedding<VOCAB, DIM, D>
{
    type Output = Tensor<Rank2<SEQ, DIM>, f32, D>;
    fn forward(&self, input: Tensor<Rank1<SEQ>, usize, D>) -> Self::Output {
        self.weight.retaped().gather(input)
    }
}

impl<
        const VOCAB: usize,
        const DIM: usize,
        const SEQ: usize,
        const BATCH: usize,
        D: Device<f32>,
    > Module<Tensor<Rank2<BATCH, SEQ>, usize, D>> for Embedding<VOCAB, DIM, D>
{
    type Output = Tensor<Rank3<BATCH, SEQ, DIM>, f32, D>;
    fn forward(&self, input: Tensor<Rank2<BATCH, SEQ>, usize, D>) -> Self::Output {
        self.weight.retaped().gather(input)
    }
}

impl<T, const VOCAB: usize, const DIM: usize, D: Device<f32>> ModuleMut<T>
    for Embedding<VOCAB, DIM, D>
where
    Self: Module<T>,
{
    type Output = <Self as Module<T>>::Output;
    fn forward_mut(&mut self, input: T) -> Self::Output {
        self.forward(input)
    }
}

impl<const VOCAB: usize, const DIM: usize, D: Device<f32>> GradientUpdate<D, f32>
    for Embedding<VOCAB, DIM, D>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), D::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.weight.update(updater, unused)?;
        Ok(())
    }
}

impl<const VOCAB: usize, const DIM: usize, D: Device<f32>> ResetParams<D, f32>
    for Embedding<VOCAB, DIM, D>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let bound: f32 = 1.0 / (VOCAB as f32).sqrt();
        let distr = rand_distr::Uniform::new(-bound, bound);
        let weight = device.try_sample(distr)?;
        Ok(Self { weight })
    }

    fn try_reset_params(&mut self) -> Result<(), D::Err> {
        let bound: f32 = 1.0 / (VOCAB as f32).sqrt();
        let distr = rand_distr::Uniform::new(-bound, bound);
        self.weight.try_fill_with_distr(distr)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{tests::SimpleUpdater, ModuleBuilder},
        tests::{assert_close, TestDevice},
        unique_id::HasUniqueId,
    };

    const W: [[f32; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];

    #[test]
    fn test_embedding_initialize() {
        let dev: TestDevice = Default::default();
        let m: Embedding<2000, 1> = dev.build_module();
        let bound = 1.0 / 2000.0f32.sqrt();
        for v in m.weight.as_vec() {
            assert!(-bound <= v && v <= bound && v != 0.0);
        }
    }

    #[test]
    fn embedding_forward_1d() {
        let dev: TestDevice = Default::default();

        let model = Embedding {
            weight: dev.tensor(W),
        };

        let x = dev.tensor([0, 0, 1]);
        let y = model.forward(x);
        assert_close(
            &y.array(),
            &[
                [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
            ],
        );

        // let g = y.square().mean().backward();
        // assert_close(
        //     &g.get(&model.weight).array(),
        //     &[
        //         [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        //         [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
        //     ],
        // );
    }

    #[test]
    fn test_forward_2d() {
        let dev: TestDevice = Default::default();

        let model = Embedding {
            weight: dev.tensor(W),
        };

        let x = dev.tensor([[0, 0], [0, 1]]);
        // let y = model.forward(x.trace());
        let y = model.forward(x);
        assert_close(
            &y.array(),
            &[
                [
                    [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                    [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                ],
                [
                    [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
                    [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
                ],
            ],
        );

        // let g = y.square().mean().backward();
        // assert_close(
        //     &g.get(&model.weight).array(),
        //     &[
        //         [-1.1541969, 0.6956873, -0.8553807, 0.9289255, 0.04931633],
        //         [0.29272807, -0.17702839, 0.08586791, -0.24057935, 0.5286576],
        //     ],
        // );
    }

    #[test]
    fn test_embedding_missing_gradients() {
        let dev: TestDevice = Default::default();

        let mut model: Embedding<5, 3> = dev.build_module();
        let mut g: SimpleUpdater = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(&unused.ids, &[*model.weight.id()]);

        g.0.try_alloc_for(&model.weight).unwrap();

        // weight gradient is present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert!(unused.is_empty());
    }
}
