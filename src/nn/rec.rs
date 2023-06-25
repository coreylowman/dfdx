use crate::{
    nn::linear::Linear,
    prelude::{
        BuildModule, BuildOnDevice, Const, Device, Dim, Dtype, HasShape, Module, ModuleVisitor,
        NonMutableModule, PutTape, SelectTo, Sigmoid, SplitTape, Tanh, Tape, Tensor,
        TensorCollection, TensorFrom, TryStack,
    },
};
use core::marker::PhantomData;
use num_traits::Float;
use rand_distr::uniform::SampleUniform;

pub mod builder {
    use crate::prelude::Dim;
    use core::marker::PhantomData;

    #[derive(Debug)]
    pub struct RNN<
        const IN_CHAN: usize,
        const OUT_CHAN: usize,
        InSeq: Dim = usize,
        OutSeq: Dim = usize,
    > {
        in_seq: PhantomData<InSeq>,
        out_seq: PhantomData<OutSeq>,
    }

    #[derive(Debug)]
    pub struct GRU<
        const IN_CHAN: usize,
        const OUT_CHAN: usize,
        InSeq: Dim = usize,
        OutSeq: Dim = usize,
    > {
        in_seq: PhantomData<InSeq>,
        out_seq: PhantomData<OutSeq>,
    }
}

impl<const I: usize, const O: usize, E, D, IS, OS> BuildOnDevice<D, E>
    for builder::RNN<I, O, IS, OS>
where
    E: Dtype,
    D: Device<E>,
    IS: Dim,
    OS: Dim,
    RNN<I, O, E, D, IS, OS>: BuildModule<D, E>,
{
    type Built = RNN<I, O, E, D, IS, OS>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

impl<const I: usize, const O: usize, E, D, IS, OS> BuildOnDevice<D, E>
    for builder::GRU<I, O, IS, OS>
where
    E: Dtype,
    D: Device<E>,
    IS: Dim,
    OS: Dim,
    GRU<I, O, E, D, IS, OS>: BuildModule<D, E>,
{
    type Built = GRU<I, O, E, D, IS, OS>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

// TODO: support batch on cell forward
pub trait RecCell<const I: usize, const O: usize, E: Dtype, D: Device<E>, T: Tape<E, D>> {
    fn cell_try_forward(
        &self,
        x: Tensor<(Const<I>,), E, D, T>,
        h: Tensor<(Const<O>,), E, D, T>,
    ) -> Result<Tensor<(Const<O>,), E, D, T>, D::Err>;
}

pub struct RNN<const I: usize, const O: usize, E: Dtype, D: Device<E>, IS: Dim, OS: Dim> {
    l_x: Linear<I, O, E, D>,
    l_h: Linear<O, O, E, D>,
    tanh: Tanh,
    is: PhantomData<IS>,
    os: PhantomData<OS>,
}

impl<const I: usize, const O: usize, E, D, IS, OS> TensorCollection<E, D>
    for RNN<I, O, E, D, IS, OS>
where
    E: Dtype + Float + SampleUniform,
    D: Device<E>,
    IS: Dim,
    OS: Dim,
{
    type To<E2: Dtype, D2: Device<E2>> = RNN<I, O, E2, D2, IS, OS>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("l_x", |s| &s.l_x, |s| &mut s.l_x),
                Self::module("l_h", |s| &s.l_h, |s| &mut s.l_h),
            ),
            |(l_x, l_h)| RNN {
                l_x,
                l_h,
                tanh: Default::default(),
                is: Default::default(),
                os: Default::default(),
            },
        )
    }
}

impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, IS: Dim, OS: Dim> NonMutableModule
    for RNN<I, O, E, D, IS, OS>
{
}

impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, T: Tape<E, D>, IS: Dim, OS: Dim>
    RecCell<I, O, E, D, T> for RNN<I, O, E, D, IS, OS>
{
    fn cell_try_forward(
        &self,
        x: Tensor<(Const<I>,), E, D, T>,
        h: Tensor<(Const<O>,), E, D, T>,
    ) -> Result<Tensor<(Const<O>,), E, D, T>, D::Err> {
        self.tanh
            .try_forward(self.l_x.try_forward(x)? + self.l_h.try_forward(h)?)
    }
}

pub struct GRU<const I: usize, const O: usize, E: Dtype, D: Device<E>, IS: Dim, OS: Dim> {
    l_xr: Linear<I, O, E, D>,
    l_hr: Linear<O, O, E, D>,
    l_xz: Linear<I, O, E, D>,
    l_hz: Linear<O, O, E, D>,
    l_xn: Linear<I, O, E, D>,
    l_hn: Linear<O, O, E, D>,
    sigmoid: Sigmoid,
    tanh: Tanh,
    is: PhantomData<IS>,
    os: PhantomData<OS>,
}

impl<const I: usize, const O: usize, E, D, IS, OS> TensorCollection<E, D>
    for GRU<I, O, E, D, IS, OS>
where
    E: Dtype + Float + SampleUniform,
    D: Device<E>,
    IS: Dim,
    OS: Dim,
{
    type To<E2: Dtype, D2: Device<E2>> = GRU<I, O, E2, D2, IS, OS>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("l_xr", |s| &s.l_xr, |s| &mut s.l_xr),
                Self::module("l_hr", |s| &s.l_hr, |s| &mut s.l_hr),
                Self::module("l_xz", |s| &s.l_xz, |s| &mut s.l_xz),
                Self::module("l_hz", |s| &s.l_hz, |s| &mut s.l_hz),
                Self::module("l_xn", |s| &s.l_xn, |s| &mut s.l_xn),
                Self::module("l_hn", |s| &s.l_hn, |s| &mut s.l_hn),
            ),
            |(l_xr, l_hr, l_xz, l_hz, l_xn, l_hn)| GRU {
                l_xr,
                l_hr,
                l_xz,
                l_hz,
                l_xn,
                l_hn,
                sigmoid: Default::default(),
                tanh: Default::default(),
                is: Default::default(),
                os: Default::default(),
            },
        )
    }
}
impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, IS: Dim, OS: Dim> NonMutableModule
    for GRU<I, O, E, D, IS, OS>
{
}
impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, T: Tape<E, D>, IS: Dim, OS: Dim>
    RecCell<I, O, E, D, T> for GRU<I, O, E, D, IS, OS>
{
    fn cell_try_forward(
        &self,
        x: Tensor<(Const<I>,), E, D, T>,
        h: Tensor<(Const<O>,), E, D, T>,
    ) -> Result<Tensor<(Const<O>,), E, D, T>, D::Err> {
        let r = self.sigmoid.try_forward(
            self.l_xr.try_forward(x.retaped::<T>())? + self.l_hr.try_forward(h.retaped::<T>())?,
        )?;
        let z = self.sigmoid.try_forward(
            self.l_xz.try_forward(x.retaped::<T>())? + self.l_hz.try_forward(h.retaped::<T>())?,
        )?;
        let n = self.tanh.try_forward(
            self.l_xn.try_forward(x)? + r * self.l_hn.try_forward(h.retaped::<T>())?,
        )?;
        let ones = D::default().ones();
        Ok((-z.retaped() + ones) * n + z * h)
    }
}

macro_rules! cell_impls {
    ($cell:ident) => {
        // usize -> usize

        impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, T: Tape<E, D>>
            Module<Tensor<(usize, Const<I>), E, D, T>> for $cell<I, O, E, D, usize, usize>
        {
            type Output = Tensor<(usize, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: usize, output_seq: usize, batch: no
            fn try_forward(
                &self,
                input: Tensor<(usize, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                let mut hs = vec![];
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for s in 0..input.shape().0 {
                    let x = input.retaped::<T>().try_select(dev.tensor(s))?;
                    let (h_new, tape_select_cell) = self.cell_try_forward(x, h)?.split_tape();
                    tape = tape.merge(tape_select_cell);
                    hs.push(h_new.retaped::<T>());
                    h = h_new.retaped::<T>();
                }
                let (hs, tape_stack) = hs.try_stack()?.split_tape();
                Ok(hs.put_tape(tape.merge(tape_stack)))
            }
        }

        impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, T: Tape<E, D>>
            Module<Tensor<(usize, usize, Const<I>), E, D, T>> for $cell<I, O, E, D, usize, usize>
        {
            type Output = Tensor<(usize, usize, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: usize, output_seq: usize, batch: usize
            fn try_forward(
                &self,
                input: Tensor<(usize, usize, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                // (batch, seq, O)
                let mut hsb = vec![];
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for b in 0..input.shape().0 {
                    // (seq, O)
                    let mut hs = vec![];
                    let seq = input.retaped::<T>().try_select(dev.tensor(b))?;
                    for s in 0..seq.shape().0 {
                        let x = seq.retaped::<T>().try_select(dev.tensor(s))?;
                        let (h_new, tape_select_cell) = self.cell_try_forward(x, h)?.split_tape();
                        tape = tape.merge(tape_select_cell);
                        hs.push(h_new.retaped::<T>());
                        h = h_new.retaped::<T>();
                    }
                    hsb.push(hs.try_stack()?);
                }
                let (hsb, tape_stack) = hsb.try_stack()?.split_tape();
                Ok(hsb.put_tape(tape.merge(tape_stack)))
            }
        }

        impl<
                const I: usize,
                const O: usize,
                const B: usize,
                E: Dtype,
                D: Device<E>,
                T: Tape<E, D>,
            > Module<Tensor<(Const<B>, usize, Const<I>), E, D, T>>
            for $cell<I, O, E, D, usize, usize>
        {
            type Output = Tensor<(Const<B>, usize, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: usize, output_seq: usize, batch: B
            fn try_forward(
                &self,
                input: Tensor<(Const<B>, usize, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                // HACK: better way of creating const size tensors
                let init = dev.zeros_like(&(input.shape().1, Const::<O>));
                // (batch, seq, O)
                let mut hsb = [(); B].map(|_| init.retaped());
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for b in 0..B {
                    // (seq, O)
                    let mut hs = vec![];
                    let seq = input.retaped::<T>().try_select(dev.tensor(b))?;
                    for s in 0..seq.shape().0 {
                        let x = seq.retaped::<T>().try_select(dev.tensor(s))?;
                        let (h_new, tape_select_cell) = self.cell_try_forward(x, h)?.split_tape();
                        tape = tape.merge(tape_select_cell);
                        hs.push(h_new.retaped::<T>());
                        h = h_new.retaped::<T>();
                    }
                    hsb[b] = hs.try_stack()?;
                }
                let (hsb, tape_stack) = hsb.try_stack()?.split_tape();
                Ok(hsb.put_tape(tape.merge(tape_stack)))
            }
        }

        // usize -> 1

        impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, T: Tape<E, D>>
            Module<Tensor<(usize, Const<I>), E, D, T>> for $cell<I, O, E, D, usize, Const<1>>
        {
            type Output = Tensor<(Const<O>,), E, D, T>;
            type Error = D::Err;

            /// input_seq: usize, output_seq: usize, batch: no
            fn try_forward(
                &self,
                input: Tensor<(usize, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                let mut h = dev.zeros().retaped::<T>();
                let (input, tape) = input.split_tape();
                for s in 0..input.shape().0 {
                    let x = input.retaped::<T>().try_select(dev.tensor(s))?;
                    h = self.cell_try_forward(x, h)?;
                }
                let (h, tape_cell) = h.split_tape();
                Ok(h.put_tape(tape.merge(tape_cell)))
            }
        }

        impl<const I: usize, const O: usize, E: Dtype, D: Device<E>, T: Tape<E, D>>
            Module<Tensor<(usize, usize, Const<I>), E, D, T>>
            for $cell<I, O, E, D, usize, Const<1>>
        {
            type Output = Tensor<(usize, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: usize, output_seq: 1, batch: usize
            fn try_forward(
                &self,
                input: Tensor<(usize, usize, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                // (batch,  O)
                let mut hb = vec![];
                let (input, tape) = input.split_tape();
                for b in 0..input.shape().0 {
                    // (seq, O)
                    let mut h = dev.zeros().retaped::<T>();
                    let seq = input.retaped::<T>().try_select(dev.tensor(b))?;
                    for s in 0..seq.shape().0 {
                        let x = seq.retaped::<T>().try_select(dev.tensor(s))?;
                        h = self.cell_try_forward(x, h)?;
                    }
                    hb.push(h);
                }
                let (hb, tape_stack) = hb.try_stack()?.split_tape();
                Ok(hb.put_tape(tape.merge(tape_stack)))
            }
        }

        impl<
                const I: usize,
                const O: usize,
                const B: usize,
                E: Dtype,
                D: Device<E>,
                T: Tape<E, D>,
            > Module<Tensor<(Const<B>, usize, Const<I>), E, D, T>>
            for $cell<I, O, E, D, usize, Const<1>>
        {
            type Output = Tensor<(Const<B>, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: usize, output_seq: usize, batch: B
            fn try_forward(
                &self,
                input: Tensor<(Const<B>, usize, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                // HACK: better way of creating const size tensors
                let init = dev.zeros_like(&(Const::<O>,));
                // (batch, seq, O)
                let mut hb = [(); B].map(|_| init.retaped::<T>());
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for b in 0..B {
                    // (seq, O)
                    let seq = input.retaped::<T>().try_select(dev.tensor(b))?;
                    for s in 0..seq.shape().0 {
                        let x = seq.retaped::<T>().try_select(dev.tensor(s))?;
                        let (h_new, tape_select_forward) =
                            self.cell_try_forward(x, h)?.split_tape();
                        tape = tape.merge(tape_select_forward);
                        h = h_new.retaped();
                    }
                    hb[b] = h.retaped::<T>();
                }
                let (hb, tape_stack) = hb.try_stack()?.split_tape();
                Ok(hb.put_tape(tape.merge(tape_stack)))
            }
        }

        // S -> S

        impl<
                const I: usize,
                const O: usize,
                const S: usize,
                E: Dtype,
                D: Device<E>,
                T: Tape<E, D>,
            > Module<Tensor<(Const<S>, Const<I>), E, D, T>>
            for $cell<I, O, E, D, Const<S>, Const<S>>
        where
            Assert<{ S > 1 }>: IsTrue,
        {
            type Output = Tensor<(Const<S>, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: S, output_seq: S, batch: no
            fn try_forward(
                &self,
                input: Tensor<(Const<S>, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                let init = dev.zeros_like(&(Const::<O>,));
                let mut hs = [(); S].map(|_| init.retaped::<T>());
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for s in 0..S {
                    let x = input.retaped::<T>().try_select(dev.tensor(s))?;
                    let (h_new, tape_select_cell) = self.cell_try_forward(x, h)?.split_tape();
                    tape = tape.merge(tape_select_cell);
                    hs[s] = h_new.retaped::<T>();
                    h = h_new.retaped::<T>();
                }
                let (hs, tape_stack) = hs.try_stack()?.split_tape();
                Ok(hs.put_tape(tape.merge(tape_stack)))
            }
        }

        impl<
                const I: usize,
                const O: usize,
                const S: usize,
                E: Dtype,
                D: Device<E>,
                T: Tape<E, D>,
            > Module<Tensor<(usize, Const<S>, Const<I>), E, D, T>>
            for $cell<I, O, E, D, Const<S>, Const<S>>
        where
            Assert<{ S > 1 }>: IsTrue,
        {
            type Output = Tensor<(usize, Const<S>, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: S, output_seq: S, batch: usize
            fn try_forward(
                &self,
                input: Tensor<(usize, Const<S>, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                // (batch, seq, O)
                let mut hsb = vec![];
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for b in 0..input.shape().0 {
                    let init = dev.zeros_like(&(Const::<O>,));
                    // (seq, O)
                    let mut hs = [(); S].map(|_| init.retaped::<T>());
                    let seq = input.retaped::<T>().try_select(dev.tensor(b))?;
                    for s in 0..S {
                        let x = seq.retaped::<T>().try_select(dev.tensor(s))?;
                        let (h_new, tape_select_cell) = self.cell_try_forward(x, h)?.split_tape();
                        tape = tape.merge(tape_select_cell);
                        hs[s] = h_new.retaped::<T>();
                        h = h_new.retaped::<T>();
                    }
                    hsb.push(hs.try_stack()?);
                }
                let (hsb, tape_stack) = hsb.try_stack()?.split_tape();
                Ok(hsb.put_tape(tape.merge(tape_stack)))
            }
        }

        impl<
                const I: usize,
                const O: usize,
                const S: usize,
                const B: usize,
                E: Dtype,
                D: Device<E>,
                T: Tape<E, D>,
            > Module<Tensor<(Const<B>, Const<S>, Const<I>), E, D, T>>
            for $cell<I, O, E, D, Const<S>, Const<S>>
        where
            Assert<{ S > 1 }>: IsTrue,
        {
            type Output = Tensor<(Const<B>, Const<S>, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: S, output_seq: S, batch: B
            fn try_forward(
                &self,
                input: Tensor<(Const<B>, Const<S>, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                // HACK: better way of creating const size tensors
                let init = dev.zeros_like(&(Const::<S>, Const::<O>));
                // (batch, seq, O)
                let mut hsb = [(); B].map(|_| init.retaped());
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for b in 0..B {
                    let init = dev.zeros_like(&(Const::<O>,));
                    // (seq, O)
                    let mut hs = [(); S].map(|_| init.retaped::<T>());
                    let seq = input.retaped::<T>().try_select(dev.tensor(b))?;
                    for s in 0..S {
                        let x = seq.retaped::<T>().try_select(dev.tensor(s))?;
                        let (h_new, tape_select_cell) = self.cell_try_forward(x, h)?.split_tape();
                        tape = tape.merge(tape_select_cell);
                        hs[s] = h_new.retaped::<T>();
                        h = h_new.retaped::<T>();
                    }
                    hsb[b] = hs.try_stack()?;
                }
                let (hsb, tape_stack) = hsb.try_stack()?.split_tape();
                Ok(hsb.put_tape(tape.merge(tape_stack)))
            }
        }

        // S -> 1

        impl<
                const I: usize,
                const O: usize,
                const S: usize,
                E: Dtype,
                D: Device<E>,
                T: Tape<E, D>,
            > Module<Tensor<(Const<S>, Const<I>), E, D, T>>
            for $cell<I, O, E, D, Const<S>, Const<1>>
        {
            type Output = Tensor<(Const<O>,), E, D, T>;
            type Error = D::Err;

            /// input_seq: S, output_seq: 1, batch: no
            fn try_forward(
                &self,
                input: Tensor<(Const<S>, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                let mut h = dev.zeros().retaped::<T>();
                let (input, tape) = input.split_tape();
                for s in 0..S {
                    let x = input.retaped::<T>().try_select(dev.tensor(s))?;
                    h = self.cell_try_forward(x, h)?;
                }
                let (h, tape_select_cell) = h.split_tape();
                Ok(h.put_tape(tape.merge(tape_select_cell)))
            }
        }

        impl<
                const I: usize,
                const O: usize,
                const S: usize,
                E: Dtype,
                D: Device<E>,
                T: Tape<E, D>,
            > Module<Tensor<(usize, Const<S>, Const<I>), E, D, T>>
            for $cell<I, O, E, D, Const<S>, Const<1>>
        {
            type Output = Tensor<(usize, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: S, output_seq: 1, batch: usize
            fn try_forward(
                &self,
                input: Tensor<(usize, Const<S>, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                // (batch, seq, O)
                let mut hb = vec![];
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for b in 0..input.shape().0 {
                    let seq = input.retaped::<T>().try_select(dev.tensor(b))?;
                    for s in 0..S {
                        let x = seq.retaped::<T>().try_select(dev.tensor(s))?;
                        let (h_new, tape_select_cell) = self.cell_try_forward(x, h)?.split_tape();
                        tape = tape.merge(tape_select_cell);
                        h = h_new.retaped::<T>();
                    }
                    hb.push(h.retaped::<T>());
                }
                let (hb, tape_stack) = hb.try_stack()?.split_tape();
                Ok(hb.put_tape(tape.merge(tape_stack)))
            }
        }

        impl<
                const I: usize,
                const O: usize,
                const S: usize,
                const B: usize,
                E: Dtype,
                D: Device<E>,
                T: Tape<E, D>,
            > Module<Tensor<(Const<B>, Const<S>, Const<I>), E, D, T>>
            for $cell<I, O, E, D, Const<S>, Const<1>>
        {
            type Output = Tensor<(Const<B>, Const<O>), E, D, T>;
            type Error = D::Err;

            /// input_seq: S, output_seq: 1, batch: B
            fn try_forward(
                &self,
                input: Tensor<(Const<B>, Const<S>, Const<I>), E, D, T>,
            ) -> Result<Self::Output, Self::Error> {
                let dev = D::default();
                // HACK: better way of creating const size tensors
                let init = dev.zeros_like(&(Const::<O>,));
                // (batch, seq, O)
                let mut hb = [(); B].map(|_| init.retaped::<T>());
                let mut h = dev.zeros().retaped::<T>();
                let (input, mut tape) = input.split_tape();
                for b in 0..B {
                    let seq = input.retaped::<T>().try_select(dev.tensor(b))?;
                    for s in 0..S {
                        let x = seq.retaped::<T>().try_select(dev.tensor(s))?;
                        let (h_new, tape_select_cell) = self.cell_try_forward(x, h)?.split_tape();
                        tape = tape.merge(tape_select_cell);
                        h = h_new.retaped::<T>();
                    }
                    hb[b] = h.retaped::<T>();
                }
                let (hsb, tape_stack) = hb.try_stack()?.split_tape();
                Ok(hsb.put_tape(tape.merge(tape_stack)))
            }
        }
    };
}

cell_impls!(RNN);
cell_impls!(GRU);

pub enum Assert<const CHECK: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

#[cfg(test)]
mod tests {
    use super::{builder::RNN, *};
    use crate::{
        prelude::{Const, DeviceBuildExt, Tensor, ZerosTensor},
        tests::{TestDevice, TestDtype},
    };

    #[test]
    fn test_forward() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros_like(&(Const::<10>, Const::<3>));
        let _: Tensor<(Const<10>, Const<1>), _, _, _> = dev
            .build_module::<RNN<3, 1, Const<10>, Const<10>>, TestDtype>()
            .forward(x.clone());
        let _: Tensor<(Const<10>, Const<5>), _, _, _> = dev
            .build_module::<RNN<3, 5, Const<10>, Const<10>>, TestDtype>()
            .forward(x.clone());
        let _: Tensor<(Const<5>,), _, _, _> = dev
            .build_module::<RNN<3, 5, Const<10>, Const<1>>, TestDtype>()
            .forward(x.clone());
    }

    #[test]
    fn test_batch_forward() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros_like(&(Const::<32>, Const::<10>, Const::<3>));
        let _: Tensor<(Const<32>, Const<10>, Const<1>), _, _, _> = dev
            .build_module::<RNN<3, 1, Const<10>, Const<10>>, TestDtype>()
            .forward(x.clone());
        let _: Tensor<(Const<32>, Const<10>, Const<5>), _, _, _> = dev
            .build_module::<RNN<3, 5, Const<10>, Const<10>>, TestDtype>()
            .forward(x.clone());
        let _: Tensor<(Const<32>, Const<5>), _, _, _> = dev
            .build_module::<RNN<3, 5, Const<10>, Const<1>>, TestDtype>()
            .forward(x.clone());
    }

    // TODO: more tests
}
