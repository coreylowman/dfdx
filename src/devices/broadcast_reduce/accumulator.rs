use super::indexing::{ElementMut, ElementRef};

pub(crate) trait Accumulator<T> {
    const INIT: T;
    fn accum(accum: &mut T, item: &T);
}

pub(crate) struct Max;
impl Accumulator<f32> for Max {
    const INIT: f32 = f32::NEG_INFINITY;
    fn accum(accum: &mut f32, item: &f32) {
        *accum = accum.max(*item);
    }
}

pub(crate) struct Min;
impl Accumulator<f32> for Min {
    const INIT: f32 = f32::INFINITY;
    fn accum(accum: &mut f32, item: &f32) {
        *accum = accum.min(*item);
    }
}

pub(crate) struct Sum;
impl Accumulator<f32> for Sum {
    const INIT: f32 = 0.0;
    fn accum(accum: &mut f32, item: &f32) {
        *accum += item;
    }
}

pub(crate) struct Mul;
impl Accumulator<f32> for Mul {
    const INIT: f32 = 1.0;
    fn accum(accum: &mut f32, item: &f32) {
        *accum *= item;
    }
}

pub(crate) struct Copy;
impl Accumulator<f32> for Copy {
    const INIT: f32 = 0.0;
    fn accum(accum: &mut f32, item: &f32) {
        *accum = *item;
    }
}

pub(super) fn accum1d<A, L, R, const M: usize>(l: &mut L, r: &R)
where
    L: ElementMut<Index = usize>,
    R: ElementRef<Index = usize, Element = L::Element>,
    A: Accumulator<L::Element>,
{
    for m in 0..M {
        A::accum(l.index_mut(m), r.index_ref(m));
    }
}

pub(super) fn accum2d<A, L, R, const M: usize, const N: usize>(l: &mut L, r: &R)
where
    L: ElementMut<Index = [usize; 2]>,
    R: ElementRef<Index = [usize; 2], Element = L::Element>,
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
    L: ElementMut<Index = [usize; 3]>,
    R: ElementRef<Index = [usize; 3], Element = L::Element>,
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
    L: ElementMut<Index = [usize; 4]>,
    R: ElementRef<Index = [usize; 4], Element = L::Element>,
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
