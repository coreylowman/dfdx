use crate::{shapes::*, tensor::*};

mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

pub trait SliceKernel<E: Unit>: DeviceStorage {
    fn forward<Src: Shape + SliceShape<Slice>, Slice>(
        &self,
        inp: &Tensor<Src, E, Self>,
        slice: &Slice,
    ) -> Result<Tensor<Src::Sliced, E, Self>, Self::Err>;

    fn backward<Src: Shape + SliceShape<Slice>, Slice>(
        &self,
        inp: &Tensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
        slice: &Slice,
    ) -> Result<(), Self::Err>;
}

impl<S: Shape, E: Unit, D: SliceKernel<E>, T: Tape<E, D>> Tensor<S, E, D, T> {
    pub fn try_slice<Slice>(self, slice: Slice) -> Result<Tensor<S::Sliced, E, D, T>, D::Err>
    where
        S: SliceShape<Slice>,
        Slice: 'static,
    {
        let (inp, mut tape) = self.split_tape();
        let out = inp.device.forward(&inp, &slice)?;
        let phantom_out = out.clone();

        tape.try_alloc_grad(&inp)?;
        tape.try_alloc_grad(&out)?;
        tape.add_backward_op(move |grads| {
            let (grad_inp, grad_out) = grads.mut_and_ref(&inp, &phantom_out);
            inp.device.backward(&inp, grad_inp, grad_out, &slice)
        });
        Ok(out.put_tape(tape))
    }

    pub fn slice<Slice>(self, slice: Slice) -> Tensor<S::Sliced, E, D, T>
    where
        S: SliceShape<Slice>,
        Slice: 'static,
    {
        self.try_slice(slice).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use crate::tests::TestDevice;

    #[test]
    fn test_slice() {
        let dev = TestDevice::default();
        let a = dev.tensor([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);

        let b: Tensor<Rank2<2, 2>, _, _> = a.clone().slice((2.., 2..)).realize().unwrap();
        assert_eq!(b.array(), [[11., 12.], [15., 16.]]);

        let b: Tensor<Rank2<2, 2>, _, _> = a.clone().slice((1..3, 1..3)).realize().unwrap();
        assert_eq!(b.array(), [[6., 7.], [10., 11.]]);

        let b: Tensor<Rank2<1, 3>, _, _> = a.clone().slice((..1, 1..4)).realize().unwrap();
        assert_eq!(b.array(), [[2., 3., 4.]]);

        let b: Tensor<Rank2<2, 3>, _, _> = a.clone().slice((1..3, ..3)).realize().unwrap();
        assert_eq!(b.array(), [[5., 6., 7.], [9., 10., 11.]]);

        let b: Tensor<Rank2<2, 3>, _, _> = a.clone().slice((1..=2, 1..=3)).realize().unwrap();
        assert_eq!(b.array(), [[6., 7., 8.], [10., 11., 12.]]);

        let b: Tensor<Rank2<2, 2>, _, _> = a.clone().slice((0..=1, 2..=3)).realize().unwrap();
        assert_eq!(b.array(), [[3., 4.], [7., 8.]]);

        let b: Tensor<Rank2<3, 2>, _, _> = a.clone().slice((1.., ..2)).realize().unwrap();
        assert_eq!(b.array(), [[5., 6.], [9., 10.], [13., 14.]]);

        let b: Tensor<Rank2<2, 2>, _, _> = a.clone().slice((..2, 2..)).realize().unwrap();
        assert_eq!(b.array(), [[3., 4.], [7., 8.]]);
    }

    #[test]
    fn test_slice_broadcast_top() {
        let dev = TestDevice::default();
        let a: Tensor<Rank2<5, 4>, _, _> = dev.tensor([1., 2., 3., 4.]).broadcast();

        let b: Tensor<Rank2<3, 4>, _, _> = a.clone().slice((..3, ..)).realize().unwrap();
        assert_eq!(b.array(), [[1., 2., 3., 4.]; 3]);

        let b: Tensor<Rank2<5, 2>, _, _> = a.clone().slice((.., 1..3)).realize().unwrap();
        assert_eq!(b.array(), [[2., 3.]; 5]);

        let b: Tensor<Rank2<2, 2>, _, _> = a.clone().slice((1..3, 1..3)).realize().unwrap();
        assert_eq!(b.array(), [[2., 3.], [2., 3.]]);

        let b: Tensor<Rank2<3, 3>, _, _> = a.clone().slice((1..4, 1..4)).realize().unwrap();
        assert_eq!(b.array(), [[2., 3., 4.]; 3]);
    }

    #[test]
    fn test_slice_broadcast_bottom() {
        let dev = TestDevice::default();
        let a: Tensor<Rank2<4, 5>, _, _> = dev.tensor([1., 2., 3., 4.]).broadcast();

        let b: Tensor<Rank2<2, 5>, _, _> = a.clone().slice((1..3, ..)).realize().unwrap();
        assert_eq!(b.array(), [[2.; 5], [3.; 5]]);

        let b: Tensor<Rank2<4, 2>, _, _> = a.clone().slice((.., 1..3)).realize().unwrap();
        assert_eq!(b.array(), [[1., 1.], [2., 2.], [3., 3.], [4., 4.]]);

        let b: Tensor<Rank2<2, 2>, _, _> = a.clone().slice((1..3, 3..)).realize().unwrap();
        assert_eq!(b.array(), [[2., 2.], [3., 3.]]);

        let b: Tensor<Rank2<2, 2>, _, _> = a.clone().slice((..2, 1..3)).realize().unwrap();
        assert_eq!(b.array(), [[1., 1.], [2., 2.]]);
    }
}
