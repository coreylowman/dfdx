mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{
    shapes::*,
    tensor::{DeviceStorage, HasErr, PutTape, SplitTape, Tape, Tensor, ZerosTensor},
};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Upscale2DOp {
    pub batch: usize,
    pub chan: usize,
    pub h_in: usize,
    pub h_out: usize,
    pub w_in: usize,
    pub w_out: usize,
}

impl Upscale2DOp {
    fn new([b, c, h_in, w_in]: [usize; 4], [h_out, w_out]: [usize; 2]) -> Self {
        assert!(
            h_out >= h_in,
            "Output height {h_out} must be larger than input height {h_in}"
        );
        assert!(
            w_out >= w_in,
            "Output width {w_out} must be larger than input width {w_in}"
        );
        Self {
            batch: b,
            chan: c,
            h_in,
            h_out,
            w_in,
            w_out,
        }
    }
}

/// Upscaling method to be used with [TryUpscale2D], can be either
/// [NearestNeighbor] or [Bilinear].
pub trait UpscaleMethod: Default {}

/// Upscales images using a pixel's nearest neighbor.
///
/// **pytorch equivalent** `F.interpolate(..., mode="nearest")`
#[derive(Clone, Copy, Default)]
pub struct NearestNeighbor;
impl UpscaleMethod for NearestNeighbor {}

/// Upscales images using bilinear interpolation between
/// a pixels neighbors
///
/// **pytorch equivalent**: `F.interpolate(..., mode="bilinear", align_corners=True)`
#[derive(Clone, Copy, Default)]
pub struct Bilinear;
impl UpscaleMethod for Bilinear {}

pub trait Upscale2DKernel<E: Unit, M: UpscaleMethod>: DeviceStorage {
    fn forward<I: Shape, O: Shape>(
        &self,
        op: Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err>;

    fn backward<I: Shape, O: Shape>(
        &self,
        op: Upscale2DOp,
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>;
}

pub trait GenericUpscale2D<M: UpscaleMethod>: HasErr {
    type Output<OH: Dim, OW: Dim>;
    fn generic_upscale2d_like<OH: Dim, OW: Dim>(
        self,
        method: M,
        height: OH,
        width: OW,
    ) -> Result<Self::Output<OH, OW>, Self::Err>;
}

/// Upscales an image to a new shape. Valid methods of upscaling are:
///
/// - [NearestNeighbor] pytorch equivalent: `F.interpolate(..., mode="nearest")`
/// - [Bilinear] pytorch equivalent: `F.interpolate(..., mode="bilinear", align_corners=True)`
///
/// Compile time upscale:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t: Tensor<Rank3<3, 32, 32>, f32, _> = dev.zeros();
/// let y: Tensor<Rank3<3, 64, 64>, f32, _> = t.upscale2d(NearestNeighbor);
/// ```
///
/// Runtime upscale:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t: Tensor<Rank3<3, 32, 32>, f32, _> = dev.zeros();
/// let y: Tensor<(Const<3>, usize, usize), f32, _> = t.upscale2d_like(NearestNeighbor, 64, 64);
/// ```
pub trait TryUpscale2D {
    /// Upscale to compile time known dimensions.
    fn upscale2d<const OH: usize, const OW: usize, M: UpscaleMethod>(
        self,
        method: M,
    ) -> <Self as GenericUpscale2D<M>>::Output<Const<OH>, Const<OW>>
    where
        Self: GenericUpscale2D<M>,
    {
        self.generic_upscale2d_like(method, Const, Const).unwrap()
    }
    /// Fallibly upscale to compile time known dimensions.
    fn try_upscale2d<const OH: usize, const OW: usize, M: UpscaleMethod>(
        self,
        method: M,
    ) -> Result<<Self as GenericUpscale2D<M>>::Output<Const<OH>, Const<OW>>, Self::Err>
    where
        Self: GenericUpscale2D<M>,
    {
        self.generic_upscale2d_like(method, Const, Const)
    }
    /// Upscale to runtime known dimensions.
    fn upscale2d_like<OH: Dim, OW: Dim, M: UpscaleMethod>(
        self,
        method: M,
        height: OH,
        width: OW,
    ) -> <Self as GenericUpscale2D<M>>::Output<OH, OW>
    where
        Self: GenericUpscale2D<M>,
    {
        self.generic_upscale2d_like(method, height, width).unwrap()
    }
    /// Fallibly upscale to runtime known dimensions.
    fn try_upscale2d_like<OH: Dim, OW: Dim, M: UpscaleMethod>(
        self,
        method: M,
        height: OH,
        width: OW,
    ) -> Result<<Self as GenericUpscale2D<M>>::Output<OH, OW>, Self::Err>
    where
        Self: GenericUpscale2D<M>,
    {
        GenericUpscale2D::generic_upscale2d_like(self, method, height, width)
    }
}
impl<S: Shape, E: Dtype, D: DeviceStorage, T> TryUpscale2D for Tensor<S, E, D, T> {}

impl<
        C: Dim,
        H: Dim,
        W: Dim,
        E: Dtype,
        M: UpscaleMethod,
        D: Upscale2DKernel<E, M> + ZerosTensor<E>,
        T: 'static + Tape<E, D>,
    > GenericUpscale2D<M> for Tensor<(C, H, W), E, D, T>
{
    type Output<OH: Dim, OW: Dim> = Tensor<(C, OH, OW), E, D, T>;

    fn generic_upscale2d_like<OH: Dim, OW: Dim>(
        self,
        _method: M,
        out_height: OH,
        out_width: OW,
    ) -> Result<Self::Output<OH, OW>, Self::Err> {
        let in_height = self.shape.1;
        let in_width = self.shape.2;

        let &(chan, _, _) = self.shape();
        let op = Upscale2DOp::new(
            [1, chan.size(), in_height.size(), in_width.size()],
            [out_height.size(), out_width.size()],
        );
        let (inp, mut tape) = self.split_tape();
        let mut out = inp.device.try_zeros_like(&(chan, out_height, out_width))?;
        inp.device.forward(op, &inp, &mut out)?;
        let inp_ghost = inp.ghost();
        let out_ghost = out.ghost();
        let out_clone = out.clone();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            inp.device
                .backward(op, &inp, grad_inp, &out_clone, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

impl<
        B: Dim,
        C: Dim,
        H: Dim,
        W: Dim,
        E: Dtype,
        M: UpscaleMethod,
        D: Upscale2DKernel<E, M> + ZerosTensor<E>,
        T: 'static + Tape<E, D>,
    > GenericUpscale2D<M> for Tensor<(B, C, H, W), E, D, T>
{
    type Output<OH: Dim, OW: Dim> = Tensor<(B, C, OH, OW), E, D, T>;

    fn generic_upscale2d_like<OH: Dim, OW: Dim>(
        self,
        _method: M,
        out_height: OH,
        out_width: OW,
    ) -> Result<Self::Output<OH, OW>, Self::Err> {
        let in_height = self.shape.2;
        let in_width = self.shape.3;

        let &(batch, chan, _, _) = self.shape();
        let op = Upscale2DOp::new(
            [batch.size(), chan.size(), in_height.size(), in_width.size()],
            [out_height.size(), out_width.size()],
        );
        let (inp, mut tape) = self.split_tape();
        let mut out = inp
            .device
            .try_zeros_like(&(batch, chan, out_height, out_width))?;
        inp.device.forward(op, &inp, &mut out)?;
        let inp_ghost = inp.ghost();
        let out_ghost = out.ghost();
        let out_clone = out.clone();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&inp_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
            inp.device
                .backward(op, &inp, grad_inp, &out_clone, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::*};

    use super::{Bilinear, NearestNeighbor, TryUpscale2D};

    #[test]
    fn test_upscale2d_nearest_even() {
        let dev = TestDevice::default();

        let x = dev.tensor([[[1.0, 0.0], [2.0, 3.0]]]);
        let y = x.leaky_trace().upscale2d::<4, 4, _>(NearestNeighbor);
        assert_close_to_literal!(
            y,
            [[
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 3.0, 3.0],
                [2.0, 2.0, 3.0, 3.0],
            ]]
        );

        let g = y.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [[[0.679570457, 0.25], [1.847264025, 5.021384231]]]
        );
    }

    #[test]
    fn test_upscale2d_nearest_uneven() {
        let dev = TestDevice::default();

        let x = dev.tensor([[[1.0, 0.0, 2.0], [2.0, 3.0, 4.0]]]);
        let y = x.leaky_trace().upscale2d::<2, 7, _>(NearestNeighbor);
        assert_close_to_literal!(
            y,
            [[
                [1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            ]]
        );

        let g = y.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [[
                [0.582488963, 0.142857143, 1.055579443],
                [1.583369164, 2.869362418, 7.799735719],
            ]]
        );
    }

    #[test]
    fn test_upscale2d_nearest_batched() {
        let dev = TestDevice::default();

        let x: Tensor<_, TestDtype, _> = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let x: Tensor<Rank3<3, 2, 3>, _, _> = [x.clone(), x.clone(), x].stack();
        let x: Tensor<Rank4<5, 3, 2, 3>, _, _> =
            [x.clone(), x.clone(), x.clone(), x.clone(), x].stack();
        let y = x.leaky_trace().upscale2d::<5, 6, _>(NearestNeighbor);
        let y_array = y.array();
        for img in y_array {
            assert_eq!(
                img,
                [[
                    [1., 1., 2., 2., 3., 3.],
                    [1., 1., 2., 2., 3., 3.],
                    [1., 1., 2., 2., 3., 3.],
                    [4., 4., 5., 5., 6., 6.],
                    [4., 4., 5., 5., 6., 6.]
                ]; 3]
            );
        }

        let grads = y.exp().mean().backward();
        assert_close_to_literal!(
            grads.get(&x),
            [[[
                [0.03624376, 0.09852076, 0.26780716],
                [0.48531687, 1.319228, 3.5860338],
            ]; 3]; 5]
        );
    }

    // Use align_corners when comparing these
    #[test]
    fn test_upscale2d_bilinear_even() {
        let dev = TestDevice::default();

        let x = dev.tensor([[[1.0, 0.0], [2.0, 3.0]]]);
        let y = x.leaky_trace().upscale2d::<4, 4, _>(Bilinear);
        assert_close_to_literal!(
            y,
            [[
                [1.0, 0.66666663, 0.33333331, 0.0],
                [1.33333325, 1.22222221, 1.11111116, 1.0],
                [1.66666675, 1.77777779, 1.88888907, 2.0],
                [2.0, 2.33333325, 2.66666651, 3.0],
            ]]
        );

        let g = y.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [[[0.8130764, 0.6928807], [1.8153939, 2.7659647]]]
        );
    }

    #[test]
    fn test_upscale2d_bilinear_uneven() {
        let dev = TestDevice::default();

        let x = dev.tensor([[[1.0, 0.0, 2.0], [2.0, 3.0, 4.0]]]);
        let y = x.leaky_trace().upscale2d::<2, 7, _>(Bilinear);
        assert_close_to_literal!(
            y,
            [[
                [1.0, 0.6666666, 0.3333333, 0.0, 0.6666667, 1.3333335, 2.0],
                [2.0, 2.3333333, 2.6666665, 3.0, 3.3333335, 3.6666667, 4.0],
            ]]
        );

        let g = y.exp().mean().backward();
        assert_close_to_literal!(
            g.get(&x),
            [[
                [0.3201411, 0.3673356, 0.7548153],
                [1.3615142, 4.6318388, 6.4302063],
            ]]
        );
    }

    #[test]
    fn test_bilinear_upscale2d_batched() {
        let dev = TestDevice::default();

        let x: Tensor<_, TestDtype, _> = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let x: Tensor<Rank3<3, 2, 3>, _, _> = [x.clone(), x.clone(), x].stack();
        let x: Tensor<Rank4<5, 3, 2, 3>, _, _> =
            [x.clone(), x.clone(), x.clone(), x.clone(), x].stack();
        let y = x.leaky_trace().upscale2d::<5, 6, _>(Bilinear);
        assert_close_to_literal!(
            y,
            [[[
                [1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
                [1.75, 2.15, 2.55, 2.95, 3.35, 3.75],
                [2.5, 2.9, 3.3, 3.7, 4.1, 4.5],
                [3.25, 3.65, 4.05, 4.45, 4.85, 5.25],
                [4.0, 4.4, 4.8, 5.2, 5.6, 6.0],
            ]; 3]; 5]
        );

        let grads = y.exp().mean().backward();
        assert_close_to_literal!(
            grads.get(&x),
            [[[
                [0.10178878, 0.30509925, 0.47953573],
                [0.42368498, 1.2699431, 1.9960163],
            ]; 3]; 5]
        );
    }
}
