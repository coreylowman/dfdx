mod cpu_kernel;

use crate::arrays::{AxesAsArray, Dtype, HasShape, PermuteShapeTo, Shape};
use crate::devices::{DeviceStorage, HasErr};
use crate::gradients::Tape;
use crate::tensor::{make_tensor, Tensor};

pub trait PermuteKernel<E: Dtype>: DeviceStorage {
    fn forward<Src: Shape, Dst: Shape<Concrete = Src::Concrete>, Axes: AxesAsArray>(
        &self,
        inp: &Self::Storage<Src, E>,
    ) -> Result<Self::Storage<Dst, E>, Self::Err>
    where
        Src: PermuteShapeTo<Dst, Axes>;
    fn backward<Src: Shape, Dst: Shape<Concrete = Src::Concrete>, Axes: AxesAsArray>(
        &self,
        inp: &Self::Storage<Src, E>,
        grad_inp: &mut Self::Storage<Src, E>,
        grad_out: &Self::Storage<Dst, E>,
    ) -> Result<(), Self::Err>
    where
        Src: PermuteShapeTo<Dst, Axes>;
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
pub trait PermuteTo<T: HasShape, Axes>: HasErr {
    fn permute(self) -> T {
        self.try_permute().unwrap()
    }
    fn try_permute(self) -> Result<T, Self::Err>;
}

impl<
        Src: Shape,
        Dst: Shape<Concrete = Src::Concrete>,
        Ax: 'static + Copy + AxesAsArray,
        E: Dtype,
        D: DeviceStorage,
        T: Tape<D>,
    > PermuteTo<Tensor<Dst, E, D, T>, Ax> for Tensor<Src, E, D, T>
where
    Src: PermuteShapeTo<Dst, Ax>,
    D: PermuteKernel<E>,
{
    fn try_permute(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        let (inp, mut tape) = self.split_tape();
        let storage = PermuteKernel::forward(&inp.device, &inp.storage)?;
        let out = make_tensor(&inp.device, storage);
        let phantom_out = out.clone();
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out)?;
            PermuteKernel::backward(&inp.device, &inp.storage, grad_inp, grad_out)?;
            Ok(())
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::arrays::{Axes2, Axes3, Axes4};
    use crate::devices::{AsArray, Cpu, Randn};
    use crate::tensor::*;
    use crate::tensor_ops::*;
    use crate::tests::build_test_device;

    #[test]
    fn test_permute_2d() {
        let dev = build_test_device!();
        let t: Tensor2D<2, 3, _> = dev.randn();
        let r: Tensor2D<3, 2, _> = t.clone().permute();
        let t_array = t.as_array();
        let r_array = r.as_array();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(r_array[j][i], t_array[i][j]);
            }
        }
    }

    #[test]
    fn test_permute_3d() {
        let dev = build_test_device!();
        let t: Tensor3D<3, 5, 7, _> = dev.randn();
        let r: Tensor3D<5, 7, 3, _> = t.clone().permute();
        let t_array = t.as_array();
        let r_array = r.as_array();
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
        let t: Tensor4D<3, 5, 7, 9, _> = dev.randn();
        let r: Tensor4D<5, 9, 3, 7, _> = t.clone().permute();
        let t_array = t.as_array();
        let r_array = r.as_array();
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
        let t: Tensor2D<3, 6, _> = dev.randn();
        let g1 = t.trace().exp().sum().backward();
        let g2 = t.trace().permute().exp().sum().backward();
        assert_eq!(g1.get(&t).as_array(), g2.get(&t).as_array());
    }

    #[test]
    fn test_permute_3d_backwards() {
        let dev = build_test_device!();
        let t: Tensor3D<3, 6, 9, _> = dev.randn();
        let g1 = t.trace().exp().sum().backward();
        let g2 = PermuteTo::<Tensor3D<6, 3, 9, _, _>, _>::permute(t.trace())
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).as_array(), g2.get(&t).as_array());
    }

    #[test]
    fn test_permute_4d_backwards() {
        let dev = build_test_device!();
        let t: Tensor4D<3, 6, 9, 11, _> = dev.randn();
        let g1 = t.trace().exp().sum().backward();
        let g2 = PermuteTo::<Tensor4D<6, 3, 11, 9, _, _>, _>::permute(t.trace())
            .exp()
            .sum()
            .backward();
        assert_eq!(g1.get(&t).as_array(), g2.get(&t).as_array());
    }

    #[test]
    fn test_valid_permutations() {
        type D = Cpu;
        // let _ = <Tensor2D<3, 5, _> as PermuteTo<_, Axes2<0, 1>>>::permute;
        let _ = <Tensor2D<3, 5, D> as PermuteTo<_, Axes2<1, 0>>>::permute;

        // let _ = <Tensor3D<3, 5, 7, _> as PermuteTo<_, Axes3<0, 1, 2>>>::permute;
        let _ = <Tensor3D<3, 5, 7, D> as PermuteTo<_, Axes3<0, 2, 1>>>::permute;
        let _ = <Tensor3D<3, 5, 7, D> as PermuteTo<_, Axes3<1, 0, 2>>>::permute;
        let _ = <Tensor3D<3, 5, 7, D> as PermuteTo<_, Axes3<1, 2, 0>>>::permute;
        let _ = <Tensor3D<3, 5, 7, D> as PermuteTo<_, Axes3<2, 0, 1>>>::permute;
        let _ = <Tensor3D<3, 5, 7, D> as PermuteTo<_, Axes3<2, 1, 0>>>::permute;

        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<0, 1, 2, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<0, 1, 3, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<0, 2, 1, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<0, 2, 3, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<0, 3, 2, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<0, 3, 1, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<1, 0, 2, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<1, 0, 3, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<1, 2, 0, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<1, 2, 3, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<1, 3, 0, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<1, 3, 2, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<2, 0, 1, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<2, 0, 3, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<2, 1, 0, 3>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<2, 1, 3, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<2, 3, 0, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<2, 3, 1, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<3, 0, 1, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<3, 0, 2, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<3, 1, 0, 2>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<3, 1, 2, 0>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<3, 2, 0, 1>>>::permute;
        let _ = <Tensor4D<3, 5, 7, 9, D> as PermuteTo<_, Axes4<3, 2, 1, 0>>>::permute;
    }
}
