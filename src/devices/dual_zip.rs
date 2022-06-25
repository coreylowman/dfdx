use super::{AllocateZeros, Cpu};
use crate::arrays::CountElements;

/// Zip two Nd arrays together and mutate both at the same time with two separate generic functions.
///
/// In general if you are modifying two things at the same time that depend on each other,
/// you need a third object for temporary storage. DualZip lets you use a single value
/// as temporary storage instead of creating a whole extra array.
///
/// E.g. if you are modifying two 3d arrays, instead of creating a third 3d array, you
/// only need 1 extra float as a temporary storage.
///
/// [ZipMapElements] doesn't allow this since it enables broadcasting.
pub trait DualZip<T: CountElements>: AllocateZeros {
    fn dual_zip<Fl, Fr>(l: &mut T, r: &mut T, fl: &mut Fl, fr: &mut Fr)
    where
        Fl: FnMut(&T::Dtype, &T::Dtype) -> T::Dtype,
        Fr: FnMut(&T::Dtype, &T::Dtype) -> T::Dtype;
}

impl DualZip<f32> for Cpu {
    fn dual_zip<Fl, Fr>(l: &mut f32, r: &mut f32, fl: &mut Fl, fr: &mut Fr)
    where
        Fl: FnMut(&f32, &f32) -> f32,
        Fr: FnMut(&f32, &f32) -> f32,
    {
        let t = fl(l, r);
        *r = fr(l, r);
        *l = t;
    }
}

impl<T: CountElements, const M: usize> DualZip<[T; M]> for Cpu
where
    Self: DualZip<T>,
{
    fn dual_zip<Fl, Fr>(l: &mut [T; M], r: &mut [T; M], fl: &mut Fl, fr: &mut Fr)
    where
        Fl: FnMut(&T::Dtype, &T::Dtype) -> T::Dtype,
        Fr: FnMut(&T::Dtype, &T::Dtype) -> T::Dtype,
    {
        for (l_i, r_i) in l.iter_mut().zip(r.iter_mut()) {
            Self::dual_zip(l_i, r_i, fl, fr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_zip_0d() {
        let mut a = 3.0;
        let mut b = 2.0;
        Cpu::dual_zip(&mut a, &mut b, &mut |a, b| a * b, &mut |a, b| a + b);
        assert_eq!(a, 6.0);
        assert_eq!(b, 5.0);
    }

    #[test]
    fn test_dual_zip_1d() {
        let mut a = [1.0, 2.0, 3.0];
        let mut b = [-4.0, 0.0, 7.0];
        Cpu::dual_zip(&mut a, &mut b, &mut |a, b| a * b, &mut |a, b| a + b);
        assert_eq!(a, [-4.0, 0.0, 21.0]);
        assert_eq!(b, [-3.0, 2.0, 10.0]);
    }

    #[test]
    fn test_dual_zip_2d() {
        let mut a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut b = [[-4.0, 0.0, 7.0], [-1.0, -1.0, -1.0]];
        Cpu::dual_zip(&mut a, &mut b, &mut |a, b| a * b, &mut |a, b| a + b);
        assert_eq!(a, [[-4.0, 0.0, 21.0], [-4.0, -5.0, -6.0]]);
        assert_eq!(b, [[-3.0, 2.0, 10.0], [3.0, 4.0, 5.0]]);
    }
}
