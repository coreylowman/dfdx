use crate::prelude::*;
use rand::Rng;
use rand_distr::Uniform;

#[derive(Default, Debug, Clone)]
pub struct Conv2D<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
> {
    pub weight: Tensor4D<OUT_CHAN, IN_CHAN, KERNEL_SIZE, KERNEL_SIZE>,
    pub bias: Tensor1D<OUT_CHAN>,
}

impl<
        const IN_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL_SIZE: usize,
        const STRIDE: usize,
        const PADDING: usize,
    > CanUpdateWithGradients for Conv2D<IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.weight.update(grads);
        self.bias.update(grads);
    }
}

impl<
        const IN_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL_SIZE: usize,
        const STRIDE: usize,
        const PADDING: usize,
    > ResetParams for Conv2D<IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING>
{
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        let k = (IN_CHAN * KERNEL_SIZE * KERNEL_SIZE) as f32;
        let bound = 1.0 / k.sqrt();
        let dist = Uniform::new(-bound, bound);
        self.weight.randomize(rng, &dist);
        self.bias.randomize(rng, &dist);
    }
}

impl<
        TAPE: 'static + Tape,
        const IN_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL_SIZE: usize,
        const STRIDE: usize,
        const PADDING: usize,
        const IN_HEIGHT: usize,
        const IN_WIDTH: usize,
    > Module<Tensor3D<IN_CHAN, IN_HEIGHT, IN_WIDTH, TAPE>>
    for Conv2D<IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING>
where
    [(); (IN_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1]:,
    [(); (IN_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1]:,
{
    type Output = Tensor3D<
        OUT_CHAN,
        { (IN_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1 },
        { (IN_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1 },
        TAPE,
    >;

    fn forward(&self, x: Tensor3D<IN_CHAN, IN_HEIGHT, IN_WIDTH, TAPE>) -> Self::Output {
        conv2d::<TAPE, IN_CHAN, OUT_CHAN, KERNEL_SIZE, STRIDE, PADDING, IN_HEIGHT, IN_WIDTH>(
            x,
            &self.weight,
            &self.bias,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_3d() {
        todo!();
    }

    #[test]
    fn test_forward_4d() {
        todo!();
    }

    #[test]
    fn test_save() {
        todo!();
    }

    #[test]
    fn test_load() {
        todo!();
    }
}
