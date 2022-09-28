use super::indexing::{IndexMut, IndexRef};

/// Accumulates sequence of values into a single value. Used
/// for reductions & broadcasts.
pub trait Accumulator<T> {
    /// The initial value to set the accumulator to.
    const INIT: T;

    fn accum(accum: &mut T, item: &T);
}

pub(crate) struct MaxAccum;
impl Accumulator<f32> for MaxAccum {
    const INIT: f32 = f32::NEG_INFINITY;
    fn accum(accum: &mut f32, item: &f32) {
        *accum = accum.max(*item);
    }
}

pub(crate) struct MinAccum;
impl Accumulator<f32> for MinAccum {
    const INIT: f32 = f32::INFINITY;
    fn accum(accum: &mut f32, item: &f32) {
        *accum = accum.min(*item);
    }
}

pub(crate) struct AddAccum;
impl Accumulator<f32> for AddAccum {
    const INIT: f32 = 0.0;
    fn accum(accum: &mut f32, item: &f32) {
        *accum += item;
    }
}

pub(crate) struct SubAccum;
impl Accumulator<f32> for SubAccum {
    const INIT: f32 = 0.0;
    fn accum(accum: &mut f32, item: &f32) {
        *accum -= item;
    }
}

pub(crate) struct MulAccum;
impl Accumulator<f32> for MulAccum {
    const INIT: f32 = 1.0;
    fn accum(accum: &mut f32, item: &f32) {
        *accum *= item;
    }
}

pub(crate) struct CopyAccum;
impl Accumulator<f32> for CopyAccum {
    const INIT: f32 = 0.0;
    fn accum(accum: &mut f32, item: &f32) {
        *accum = *item;
    }
}

pub(crate) struct EqAccum;
impl Accumulator<f32> for EqAccum {
    const INIT: f32 = 0.0;
    fn accum(accum: &mut f32, item: &f32) {
        *accum = if accum == item { 1.0 } else { 0.0 };
    }
}

pub(super) fn accum1d<A, L, R, const M: usize>(l: &mut L, r: &R)
where
    L: IndexMut<Index = usize>,
    R: IndexRef<Index = usize, Element = L::Element>,
    A: Accumulator<L::Element>,
{
    for m in 0..M {
        A::accum(l.index_mut(m), r.index_ref(m));
    }
}

pub(super) fn accum2d<A, L, R, const M: usize, const N: usize>(l: &mut L, r: &R)
where
    L: IndexMut<Index = [usize; 2]>,
    R: IndexRef<Index = [usize; 2], Element = L::Element>,
    A: Accumulator<L::Element>,
{
    for m in 0..M {
        for n in 0..N {
            A::accum(l.index_mut([m, n]), r.index_ref([m, n]));
        }
    }
}

pub(super) fn accum3d<A, L, R, const M: usize, const N: usize, const O: usize>(l: &mut L, r: &R)
where
    L: IndexMut<Index = [usize; 3]>,
    R: IndexRef<Index = [usize; 3], Element = L::Element>,
    A: Accumulator<L::Element>,
{
    for m in 0..M {
        for n in 0..N {
            for o in 0..O {
                A::accum(l.index_mut([m, n, o]), r.index_ref([m, n, o]));
            }
        }
    }
}

pub(super) fn accum4d<A, L, R, const M: usize, const N: usize, const O: usize, const P: usize>(
    l: &mut L,
    r: &R,
) where
    L: IndexMut<Index = [usize; 4]>,
    R: IndexRef<Index = [usize; 4], Element = L::Element>,
    A: Accumulator<L::Element>,
{
    for m in 0..M {
        for n in 0..N {
            for o in 0..O {
                for p in 0..P {
                    A::accum(l.index_mut([m, n, o, p]), r.index_ref([m, n, o, p]));
                }
            }
        }
    }
}
