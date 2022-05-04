use super::ZipMapElements;

pub trait AddElements<Rhs>: Sized + ZipMapElements<Rhs> {
    fn add(&self, rhs: &Rhs) -> Self {
        self.zip_map(rhs, |x, y| x + y)
    }
    fn add_assign(&mut self, rhs: &Rhs) {
        self.zip_map_assign(rhs, |x, y| *x += y)
    }
}

impl<T, Rhs> AddElements<Rhs> for T where T: ZipMapElements<Rhs> {}

pub trait SubElements<Rhs>: Sized + ZipMapElements<Rhs> {
    fn sub(&self, rhs: &Rhs) -> Self {
        self.zip_map(rhs, |x, y| x - y)
    }
    fn sub_assign(&mut self, rhs: &Rhs) {
        self.zip_map_assign(rhs, |x, y| *x -= y);
    }
}

impl<T, Rhs> SubElements<Rhs> for T where T: ZipMapElements<Rhs> {}

pub trait MulElements<Rhs>: Sized + ZipMapElements<Rhs> {
    fn mul(&self, rhs: &Rhs) -> Self {
        self.zip_map(rhs, |x, y| x * y)
    }
    fn mul_assign(&mut self, rhs: &Rhs) {
        self.zip_map_assign(rhs, |x, y| *x *= y);
    }
}
impl<T, Rhs> MulElements<Rhs> for T where T: ZipMapElements<Rhs> {}

pub trait DivElements<Rhs>: Sized + ZipMapElements<Rhs> {
    fn div(&self, rhs: &Rhs) -> Self {
        self.zip_map(rhs, |x, y| x / y)
    }
    fn div_assign(&mut self, rhs: &Rhs) {
        self.zip_map_assign(rhs, |x, y| *x /= y);
    }
}

impl<T, Rhs> DivElements<Rhs> for T where T: ZipMapElements<Rhs> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_arith() {
        assert_eq!(AddElements::add(&1.0, &2.0), 3.0);
        assert_eq!(SubElements::sub(&1.0, &2.0), -1.0);
        assert_eq!(MulElements::mul(&1.0, &2.0), 2.0);
        assert_eq!(DivElements::div(&1.0, &2.0), 0.5);
    }

    #[test]
    fn test_1d_arith() {
        assert_eq!([1.0; 5].add(&[2.0; 5]), [3.0; 5]);
        assert_eq!([1.0; 5].add(&2.0), [3.0; 5]);

        assert_eq!([1.0; 5].sub(&[2.0; 5]), [-1.0; 5]);
        assert_eq!([1.0; 5].sub(&2.0), [-1.0; 5]);

        assert_eq!([1.0; 5].mul(&[2.0; 5]), [2.0; 5]);
        assert_eq!([1.0; 5].mul(&2.0), [2.0; 5]);

        assert_eq!([1.0; 5].div(&[2.0; 5]), [0.5; 5]);
        assert_eq!([1.0; 5].div(&2.0), [0.5; 5]);
    }

    #[test]
    fn test_2d_arith() {
        assert_eq!([[1.0; 3]; 5].add(&[[2.0; 3]; 5]), [[3.0; 3]; 5]);
        assert_eq!([[1.0; 3]; 5].add(&[2.0; 5]), [[3.0; 3]; 5]);

        assert_eq!([[1.0; 3]; 5].sub(&[[2.0; 3]; 5]), [[-1.0; 3]; 5]);
        assert_eq!([[1.0; 3]; 5].sub(&[2.0; 5]), [[-1.0; 3]; 5]);

        assert_eq!([[1.0; 3]; 5].mul(&[[2.0; 3]; 5]), [[2.0; 3]; 5]);
        assert_eq!([[1.0; 3]; 5].mul(&[2.0; 5]), [[2.0; 3]; 5]);

        assert_eq!([[1.0; 3]; 5].div(&[[2.0; 3]; 5]), [[0.5; 3]; 5]);
        assert_eq!([[1.0; 3]; 5].div(&[2.0; 5]), [[0.5; 3]; 5]);
    }

    #[test]
    fn test_3d_arith() {
        assert_eq!(
            [[[1.0; 2]; 3]; 5].add(&[[[2.0; 2]; 3]; 5]),
            [[[3.0; 2]; 3]; 5]
        );
        assert_eq!([[[1.0; 2]; 3]; 5].add(&[[2.0; 3]; 5]), [[[3.0; 2]; 3]; 5]);

        assert_eq!(
            [[[1.0; 2]; 3]; 5].sub(&[[[2.0; 2]; 3]; 5]),
            [[[-1.0; 2]; 3]; 5]
        );
        assert_eq!([[[1.0; 2]; 3]; 5].sub(&[[2.0; 3]; 5]), [[[-1.0; 2]; 3]; 5]);

        assert_eq!(
            [[[1.0; 2]; 3]; 5].mul(&[[[2.0; 2]; 3]; 5]),
            [[[2.0; 2]; 3]; 5]
        );
        assert_eq!([[[1.0; 2]; 3]; 5].mul(&[[2.0; 3]; 5]), [[[2.0; 2]; 3]; 5]);

        assert_eq!(
            [[[1.0; 2]; 3]; 5].div(&[[[2.0; 2]; 3]; 5]),
            [[[0.5; 2]; 3]; 5]
        );
        assert_eq!([[[1.0; 2]; 3]; 5].div(&[[2.0; 3]; 5]), [[[0.5; 2]; 3]; 5]);
    }
}
