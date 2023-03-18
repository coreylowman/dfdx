mod cpu_kernel;
#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::prelude::{DeviceStorage, Shape, Tensor, Unit};

pub trait ToDtypeKernel<E1: Unit, E2: Unit>: DeviceStorage {
    fn forward<S: Shape>(inp: Tensor<S, E1, Self>) -> Result<Tensor<S, E2, Self>, Self::Err>;
}

impl<S: Shape, E: Unit, D: DeviceStorage> Tensor<S, E, D> {
    pub fn try_to_dtype<E2: Unit>(self) -> Result<Tensor<S, E2, D>, D::Err>
    where
        D: ToDtypeKernel<E, E2>,
    {
        D::forward(self)
    }

    pub fn to_dtype<E2: Unit>(self) -> Tensor<S, E2, D>
    where
        D: ToDtypeKernel<E, E2>,
    {
        self.try_to_dtype().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor::*, tests::*};

    #[test]
    fn test_to_dtype() {
        let dev: TestDevice = Default::default();
        let a: Tensor<_, f32, _> = dev.tensor([1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = a.clone().to_dtype::<f32>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<f64>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<u8>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<u16>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<u32>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<u64>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<usize>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<i8>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<i16>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<i32>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<i64>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());
        let b = a.clone().to_dtype::<isize>().to_dtype::<f32>();
        assert_eq!(a.array(), b.array());

        let a: Tensor<_, bool, _> = dev.tensor([true, true, false, true, false]);
        let b = a.to_dtype::<usize>();
        assert_eq!(b.array(), [1, 1, 0, 1, 0]);
    }
}
