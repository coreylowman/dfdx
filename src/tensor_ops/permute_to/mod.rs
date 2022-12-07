mod cpu_kernel;

use crate::gradients::Tape;
use crate::shapes::{Axes, Dtype, HasShape, PermuteShapeTo, Shape};
use crate::tensor::storage::{DeviceStorage, HasErr};
use crate::tensor::{PutTape, SplitTape, Tensor, TensorFromStorage};

pub trait PermuteKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>;
    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: PermuteShapeTo<Dst, Ax>;
}

/// Permutes self into `T` with the new order of axes specified via `Axes`.
/// Permutes the tensor
///
/// Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let _: Tensor2D<3, 2> = Tensor2D::<2, 3>::zeros().permute();
/// let _: Tensor3D<3, 4, 2> = Tensor3D::<2, 3, 4>::zeros().permute();
/// let _: Tensor4D<3, 4, 5, 2> = Tensor4D::<2, 3, 4, 5>::zeros().permute();
/// ```
pub trait PermuteTo: HasErr + HasShape {
    fn permute<Dst: Shape, Ax: Axes>(self) -> Self::WithShape<Dst>
    where
        Self::Shape: PermuteShapeTo<Dst, Ax>,
    {
        self.try_permute().unwrap()
    }
    fn try_permute<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: PermuteShapeTo<Dst, Ax>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage + PermuteKernel<E>, T: Tape<D>> PermuteTo
    for Tensor<S, E, D, T>
{
    fn try_permute<Dst: Shape, Ax: Axes>(self) -> Result<Self::WithShape<Dst>, Self::Err>
    where
        Self::Shape: PermuteShapeTo<Dst, Ax>,
    {
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.upgrade(inp.device.forward(&inp.storage)?);
        let phantom_out = out.clone();
        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device.backward(grad_inp, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::shapes::{Axes2, Axes3, Axes4, Rank2, Rank3, Rank4};
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_permute_2d() {
        let dev = build_test_device!();
        let t: Tensor2D<2, 3, _> = dev.randn();
        let r: Tensor2D<3, 2, _> = t.clone().permute();
        let t_array = t.array();
        let r_array = r.array();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(r_array[j][i], t_array[i][j]);
            }
        }
    }

    #[test]
    fn test_permute_3d() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<3, 5, 7>>();
        let r = t.clone().permute::<Rank3<5, 7, 3>, _>();
        let t_array = t.array();
        let r_array = r.array();
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    assert_eq!(r_array[j][k][i], t_array[i][j][k]);
                }
            }
        }
    }

    #[test]
    fn test_permute_4d() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank4<3, 5, 7, 9>>();
        let r = t.clone().permute::<Rank4<5, 9, 3, 7>, _>();
        let t_array = t.array();
        let r_array = r.array();
        for i in 0..3 {
            for j in 0..5 {
                for k in 0..7 {
                    for l in 0..9 {
                        assert_eq!(r_array[j][l][i][k], t_array[i][j][k][l]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_permute_2d_backwards() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank2<3, 6>>();
        let g1 = t.trace().exp().sum().backward();
        let g2 = t.trace().permute().exp().sum().backward();
        assert_eq!(g1.get(&t).array(), g2.get(&t).array());
    }

    #[test]
    fn test_permute_3d_backwards() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank3<3, 6, 9>>();
        let g1 = t.trace().exp().sum().backward();
        let g2 = t
            .trace()
            .permute::<Rank3<6, 3, 9>, _>()
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).array(), g2.get(&t).array());
    }

    #[test]
    fn test_permute_4d_backwards() {
        let dev = build_test_device!();
        let t = dev.randn::<Rank4<3, 6, 9, 11>>();
        let g1 = t.trace().exp().sum().backward();
        let g2 = t
            .trace()
            .permute::<Rank4<6, 3, 11, 9>, _>()
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).array(), g2.get(&t).array());
    }

    #[test]
    fn test_valid_permutations() {
        let dev = build_test_device!();

        let _ = dev.randn::<Rank2<3, 5>>().permute::<_, Axes2<1, 0>>();

        let _ = dev.randn::<Rank3<3, 5, 7>>().permute::<_, Axes3<0, 2, 1>>();
        let _ = dev.randn::<Rank3<3, 5, 7>>().permute::<_, Axes3<1, 0, 2>>();
        let _ = dev.randn::<Rank3<3, 5, 7>>().permute::<_, Axes3<1, 2, 0>>();
        let _ = dev.randn::<Rank3<3, 5, 7>>().permute::<_, Axes3<2, 0, 1>>();
        let _ = dev.randn::<Rank3<3, 5, 7>>().permute::<_, Axes3<2, 1, 0>>();

        let x = dev.randn::<Rank4<3, 5, 7, 9>>();
        x.clone().permute::<_, Axes4<0, 1, 2, 3>>();
        x.clone().permute::<_, Axes4<0, 1, 3, 2>>();
        x.clone().permute::<_, Axes4<0, 2, 1, 3>>();
        x.clone().permute::<_, Axes4<0, 2, 3, 1>>();
        x.clone().permute::<_, Axes4<0, 3, 2, 1>>();
        x.clone().permute::<_, Axes4<0, 3, 1, 2>>();
        x.clone().permute::<_, Axes4<1, 0, 2, 3>>();
        x.clone().permute::<_, Axes4<1, 0, 3, 2>>();
        x.clone().permute::<_, Axes4<1, 2, 0, 3>>();
        x.clone().permute::<_, Axes4<1, 2, 3, 0>>();
        x.clone().permute::<_, Axes4<1, 3, 0, 2>>();
        x.clone().permute::<_, Axes4<1, 3, 2, 0>>();
        x.clone().permute::<_, Axes4<2, 0, 1, 3>>();
        x.clone().permute::<_, Axes4<2, 0, 3, 1>>();
        x.clone().permute::<_, Axes4<2, 1, 0, 3>>();
        x.clone().permute::<_, Axes4<2, 1, 3, 0>>();
        x.clone().permute::<_, Axes4<2, 3, 0, 1>>();
        x.clone().permute::<_, Axes4<2, 3, 1, 0>>();
        x.clone().permute::<_, Axes4<3, 0, 1, 2>>();
        x.clone().permute::<_, Axes4<3, 0, 2, 1>>();
        x.clone().permute::<_, Axes4<3, 1, 0, 2>>();
        x.clone().permute::<_, Axes4<3, 1, 2, 0>>();
        x.clone().permute::<_, Axes4<3, 2, 0, 1>>();
        x.permute::<_, Axes4<3, 2, 1, 0>>();
    }
}
