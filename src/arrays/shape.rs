pub trait Dtype:
    'static + Copy + Clone + std::fmt::Debug + Default + PartialOrd + Send + Sync
{
}
impl Dtype for f32 {}
impl Dtype for f64 {}
impl Dtype for usize {}

pub trait Dim: 'static + Copy + Clone + std::fmt::Debug + Send + Sync {
    fn size(&self) -> usize;
    fn from_size(size: usize) -> Option<Self>;
}

#[derive(Clone, Copy, Debug)]
pub struct Dyn(pub usize);

impl Dim for Dyn {
    #[inline(always)]
    fn size(&self) -> usize {
        self.0
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        Some(Self(size))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct C<const M: usize>;
impl<const M: usize> Dim for C<M> {
    #[inline(always)]
    fn size(&self) -> usize {
        M
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == M {
            Some(C)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StridesFor<Sh: Shape>(pub(crate) Sh::Concrete);

pub trait Shape: 'static + std::fmt::Debug + Clone + Copy + Send + Sync {
    const NUM_DIMS: usize;
    type Concrete: std::fmt::Debug
        + Clone
        + Copy
        + Default
        + std::ops::Index<usize, Output = usize>
        + std::ops::IndexMut<usize>
        + Send
        + Sync
        + IntoIterator<Item = usize>;

    #[inline(always)]
    fn num_elements(&self) -> usize {
        self.concrete().into_iter().product()
    }

    fn concrete(&self) -> Self::Concrete;
    fn strides(&self) -> StridesFor<Self>;
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self>;
}

pub trait HasShape {
    type Shape: Shape;
    fn shape(&self) -> &Self::Shape;
}

pub trait HasDtype {
    type Dtype: Dtype;
}

pub trait TryFromNumElements: Shape {
    fn try_from_num_elements(num_elements: usize) -> Option<Self>;
}

pub type Rank0 = ();
pub type Rank1<const M: usize> = (C<M>,);
pub type Rank2<const M: usize, const N: usize> = (C<M>, C<N>);
pub type Rank3<const M: usize, const N: usize, const O: usize> = (C<M>, C<N>, C<O>);
pub type Rank4<const M: usize, const N: usize, const O: usize, const P: usize> =
    (C<M>, C<N>, C<O>, C<P>);
pub type Rank5<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize> =
    (C<M>, C<N>, C<O>, C<P>, C<Q>);
pub type Rank6<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    const R: usize,
> = (C<M>, C<N>, C<O>, C<P>, C<Q>, C<R>);

impl Shape for () {
    const NUM_DIMS: usize = 0;
    type Concrete = [usize; 0];
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        []
    }
    #[inline(always)]
    fn strides(&self) -> StridesFor<Self> {
        StridesFor([])
    }
    #[inline(always)]
    fn from_concrete(_: &Self::Concrete) -> Option<Self> {
        Some(())
    }
}

impl TryFromNumElements for () {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        if num_elements == 1 {
            Some(())
        } else {
            None
        }
    }
}

impl<D1: Dim> Shape for (D1,) {
    const NUM_DIMS: usize = 1;
    type Concrete = [usize; 1];
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        [self.0.size()]
    }
    #[inline(always)]
    fn strides(&self) -> StridesFor<Self> {
        StridesFor([1])
    }
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
        let d1 = D1::from_size(concrete[0])?;
        Some((d1,))
    }
}

impl<const M: usize> TryFromNumElements for (C<M>,) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        if num_elements == M {
            Some(Default::default())
        } else {
            None
        }
    }
}

impl TryFromNumElements for (Dyn,) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        Some((Dyn(num_elements),))
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, D2) {
    const NUM_DIMS: usize = 2;
    type Concrete = [usize; 2];
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        [self.0.size(), self.1.size()]
    }
    #[inline(always)]
    fn strides(&self) -> StridesFor<Self> {
        StridesFor([self.1.size(), 1])
    }
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
        let d1 = D1::from_size(concrete[0])?;
        let d2 = D2::from_size(concrete[1])?;
        Some((d1, d2))
    }
}

impl<const M: usize, const N: usize> TryFromNumElements for (C<M>, C<N>) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        let shape: Self = Default::default();
        if shape.num_elements() == num_elements {
            Some(shape)
        } else {
            None
        }
    }
}

impl<const N: usize> TryFromNumElements for (Dyn, C<N>) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        if num_elements % N == 0 {
            Some((Dyn(num_elements / N), C))
        } else {
            None
        }
    }
}

impl<const M: usize> TryFromNumElements for (C<M>, Dyn) {
    fn try_from_num_elements(num_elements: usize) -> Option<Self> {
        if num_elements % M == 0 {
            Some((C, Dyn(num_elements / M)))
        } else {
            None
        }
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Shape for (D1, D2, D3) {
    const NUM_DIMS: usize = 3;
    type Concrete = [usize; 3];
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        [self.0.size(), self.1.size(), self.2.size()]
    }
    #[inline(always)]
    fn strides(&self) -> StridesFor<Self> {
        let a = 1;
        let b = a * self.2.size();
        let c = b * self.1.size();
        StridesFor([c, b, a])
    }
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
        let d1 = D1::from_size(concrete[0])?;
        let d2 = D2::from_size(concrete[1])?;
        let d3 = D3::from_size(concrete[2])?;
        Some((d1, d2, d3))
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> Shape for (D1, D2, D3, D4) {
    const NUM_DIMS: usize = 4;
    type Concrete = [usize; 4];
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        [self.0.size(), self.1.size(), self.2.size(), self.3.size()]
    }
    #[inline(always)]
    fn strides(&self) -> StridesFor<Self> {
        let a = 1;
        let b = a * self.3.size();
        let c = b * self.2.size();
        let d = c * self.1.size();
        StridesFor([d, c, b, a])
    }
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
        let d1 = D1::from_size(concrete[0])?;
        let d2 = D2::from_size(concrete[1])?;
        let d3 = D3::from_size(concrete[2])?;
        let d4 = D4::from_size(concrete[3])?;
        Some((d1, d2, d3, d4))
    }
}
impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> Shape for (D1, D2, D3, D4, D5) {
    const NUM_DIMS: usize = 5;
    type Concrete = [usize; 5];
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        [
            self.0.size(),
            self.1.size(),
            self.2.size(),
            self.3.size(),
            self.4.size(),
        ]
    }
    #[inline(always)]
    fn strides(&self) -> StridesFor<Self> {
        let a = 1;
        let b = a * self.4.size();
        let c = b * self.3.size();
        let d = c * self.2.size();
        let e = d * self.1.size();
        StridesFor([e, d, c, b, a])
    }
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
        let d1 = D1::from_size(concrete[0])?;
        let d2 = D2::from_size(concrete[1])?;
        let d3 = D3::from_size(concrete[2])?;
        let d4 = D4::from_size(concrete[3])?;
        let d5 = D5::from_size(concrete[4])?;
        Some((d1, d2, d3, d4, d5))
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim> Shape for (D1, D2, D3, D4, D5, D6) {
    const NUM_DIMS: usize = 6;
    type Concrete = [usize; 6];
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        [
            self.0.size(),
            self.1.size(),
            self.2.size(),
            self.3.size(),
            self.4.size(),
            self.5.size(),
        ]
    }
    #[inline(always)]
    fn strides(&self) -> StridesFor<Self> {
        let a = 1;
        let b = a * self.5.size();
        let c = b * self.4.size();
        let d = c * self.3.size();
        let e = d * self.2.size();
        let f = e * self.1.size();
        StridesFor([f, e, d, c, b, a])
    }
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
        let d1 = D1::from_size(concrete[0])?;
        let d2 = D2::from_size(concrete[1])?;
        let d3 = D3::from_size(concrete[2])?;
        let d4 = D4::from_size(concrete[3])?;
        let d5 = D5::from_size(concrete[4])?;
        let d6 = D6::from_size(concrete[5])?;
        Some((d1, d2, d3, d4, d5, d6))
    }
}
