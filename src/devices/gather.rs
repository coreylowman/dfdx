use super::{AllocateZeros, Broadcast, BroadcastMut, Cpu};
use crate::arrays::{CountElements, MultiDimensional};

pub trait ForEachLast<O: CountElements, L: CountElements + MultiDimensional, R: CountElements> {
    fn foreachlast_mb<F>(l: &mut L, r: Broadcast<R>, f: &mut F)
    where
        F: FnMut(&mut L::LastDim, &R::Dtype);
    fn foreachlast_mrb<F>(o: BroadcastMut<O>, l: &L, r: Broadcast<R>, f: &mut F)
    where
        F: FnMut(&mut O::Dtype, &L::LastDim, &R::Dtype);
}

impl<const M: usize> ForEachLast<f32, [f32; M], usize> for Cpu {
    fn foreachlast_mb<F>(l: &mut [f32; M], r: Broadcast<usize>, f: &mut F)
    where
        F: FnMut(&mut [f32; M], &usize),
    {
        f(l, r.0);
    }

    fn foreachlast_mrb<F>(o: BroadcastMut<f32>, l: &[f32; M], r: Broadcast<usize>, f: &mut F)
    where
        F: FnMut(&mut f32, &[f32; M], &usize),
    {
        f(o.0, l, r.0);
    }
}

impl<O: CountElements, L: CountElements + MultiDimensional, R: CountElements, const M: usize>
    ForEachLast<[O; M], [L; M], [R; M]> for Cpu
where
    Cpu: ForEachLast<O, L, R>,
{
    fn foreachlast_mb<F>(l: &mut [L; M], r: Broadcast<[R; M]>, f: &mut F)
    where
        F: FnMut(&mut L::LastDim, &<[R; M] as CountElements>::Dtype),
    {
        for (l_i, r_i) in l.iter_mut().zip(r.0.iter()) {
            Self::foreachlast_mb(l_i, Broadcast(r_i), f);
        }
    }

    fn foreachlast_mrb<F>(o: BroadcastMut<[O; M]>, l: &[L; M], r: Broadcast<[R; M]>, f: &mut F)
    where
        F: FnMut(&mut O::Dtype, &L::LastDim, &R::Dtype),
    {
        for i in 0..M {
            Self::foreachlast_mrb(BroadcastMut(&mut o.0[i]), &l[i], Broadcast(&r.0[i]), f)
        }
    }
}

/// Something that can be indexed by a usize array ([GatherElements::Indices]).
///
/// Output type is [GatherElements::Gathered].
pub trait GatherElements<T: CountElements>: AllocateZeros {
    type Indices: CountElements<Dtype = usize>;
    type Gathered: CountElements<Dtype = T::Dtype>;

    fn gather_into(inp: &T, indices: &Self::Indices, out: &mut Self::Gathered);

    fn gather(inp: &T, indices: &Self::Indices) -> Box<Self::Gathered> {
        let mut out = Self::zeros();
        Self::gather_into(inp, indices, &mut out);
        out
    }
}

impl GatherElements<f32> for Cpu {
    type Indices = usize;
    type Gathered = f32;

    fn gather_into(inp: &f32, _: &Self::Indices, out: &mut Self::Gathered) {
        *out = *inp;
    }
}

impl<const M: usize> GatherElements<[f32; M]> for Cpu {
    type Indices = usize;
    type Gathered = f32;

    fn gather_into(inp: &[f32; M], indices: &Self::Indices, out: &mut Self::Gathered) {
        *out = inp[*indices];
    }
}

impl<const M: usize, const N: usize> GatherElements<[[f32; N]; M]> for Cpu {
    type Indices = [usize; M];
    type Gathered = [f32; M];

    fn gather_into(inp: &[[f32; N]; M], indices: &Self::Indices, out: &mut Self::Gathered) {
        for i in 0..M {
            Self::gather_into(&inp[i], &indices[i], &mut out[i]);
        }
    }
}

impl<const M: usize, const N: usize, const O: usize> GatherElements<[[[f32; O]; N]; M]> for Cpu {
    type Indices = [[usize; N]; M];
    type Gathered = [[f32; N]; M];

    fn gather_into(inp: &[[[f32; O]; N]; M], indices: &Self::Indices, out: &mut Self::Gathered) {
        for i in 0..M {
            Self::gather_into(&inp[i], &indices[i], &mut out[i]);
        }
    }
}

impl<const M: usize, const N: usize, const O: usize, const P: usize>
    GatherElements<[[[[f32; P]; O]; N]; M]> for Cpu
{
    type Indices = [[[usize; O]; N]; M];
    type Gathered = [[[f32; O]; N]; M];

    fn gather_into(
        inp: &[[[[f32; P]; O]; N]; M],
        indices: &Self::Indices,
        out: &mut Self::Gathered,
    ) {
        for i in 0..M {
            Self::gather_into(&inp[i], &indices[i], &mut out[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_foreachlast() {
        let l = [1.0, 2.0, 3.0];
        for i in 0..3 {
            let mut o = 0.0;
            Cpu::foreachlast_mrb(BroadcastMut(&mut o), &l, Broadcast(&i), &mut |a, b, c| {
                *a = b[*c];
            });
            assert_eq!(o, l[i]);
        }
    }

    #[test]
    fn test_2d_foreachlast() {
        let l = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let r = [2, 0, 1];
        let mut o = [0.0; 3];
        Cpu::foreachlast_mrb(BroadcastMut(&mut o), &l, Broadcast(&r), &mut |a, b, c| {
            *a = b[*c];
        });
        assert_eq!(o, [3.0, 4.0, 8.0]);
    }

    #[test]
    fn test_1d_gather() {
        let t = [0.0, 1.0, 2.0];
        let i = 1;
        let mut out = 0.0;
        Cpu::gather_into(&t, &i, &mut out);
        assert_eq!(out, 1.0);
    }

    #[test]
    fn test_2d_gather() {
        let t = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
        let i = [1, 2, 0];
        let mut out = [0.0, 0.0, 0.0];
        Cpu::gather_into(&t, &i, &mut out);
        assert_eq!(out, [1.0, 5.0, 6.0]);
    }

    #[test]
    fn test_3d_gather() {
        let t = [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0], [-6.0, -7.0, -8.0]],
        ];
        let i = [[2, 1, 0], [1, 0, 1]];
        let mut out = [[0.0; 3]; 2];
        Cpu::gather_into(&t, &i, &mut out);
        assert_eq!(out, [[2.0, 4.0, 6.0], [-1.0, -3.0, -7.0]]);
    }
}
