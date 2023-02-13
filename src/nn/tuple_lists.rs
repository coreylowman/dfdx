use std::vec::Vec;

use crate::prelude::{DeviceStorage, Dtype, HasErr, HasShape, Shape, Tensor};

pub type TensorNames<A> = <A as GetTensors>::Names;
pub type Tensors<'a, A> = <A as GetTensors>::Tensors<'a>;
pub type TensorsMut<'a, A> = <A as GetTensors>::TensorsMut<'a>;
pub type TupleAppend<A, B> = <A as TupleList>::Appended<B>;
pub type AppendNames<A, B> = TupleAppend<TensorNames<A>, TensorNames<B>>;
pub type AppendTensors<'a, A, B> = TupleAppend<Tensors<'a, A>, Tensors<'a, B>>;
pub type AppendTensorsMut<'a, A, B> = TupleAppend<TensorsMut<'a, A>, TensorsMut<'a, B>>;

pub struct TupleVec<H: TupleList, T: TupleList>(pub Vec<H>, pub T);

pub trait TupleList {
    type Appended<T: TupleList>: TupleList;

    fn cons<T>(self, elem: T) -> (T, Self)
    where
        Self: Sized,
    {
        (elem, self)
    }

    fn append<T: TupleList>(self, list: T) -> Self::Appended<T>;
}

impl TupleList for () {
    type Appended<T: TupleList> = T;

    fn append<T: TupleList>(self, list: T) -> Self::Appended<T> {
        list
    }
}

impl<H1, T1: TupleList> TupleList for (H1, T1) {
    type Appended<T: TupleList> = (H1, TupleAppend<T1, T>);

    fn append<T: TupleList>(self, list: T) -> Self::Appended<T> {
        (self.0, self.1.append(list))
    }
}

impl<H1: TupleList, T1: TupleList> TupleList for TupleVec<H1, T1> {
    type Appended<T: TupleList> = TupleVec<H1, TupleAppend<T1, T>>;

    fn append<T: TupleList>(self, list: T) -> Self::Appended<T> {
        TupleVec(self.0, self.1.append(list))
    }
}

pub trait TupleZip<T>: TupleList {
    type Zipped: TupleList;

    fn zip(self, other: T) -> Option<Self::Zipped>;
}

impl TupleZip<()> for () {
    type Zipped = ();

    fn zip(self, _other: ()) -> Option<()> {
        Some(())
    }
}

impl<H1, T1, H2, T2> TupleZip<(H2, T2)> for (H1, T1)
where
    T1: TupleZip<T2>,
    T2: TupleList,
{
    type Zipped = ((H1, H2), <T1 as TupleZip<T2>>::Zipped);

    fn zip(self, other: (H2, T2)) -> Option<Self::Zipped> {
        Some(((self.0, other.0), self.1.zip(other.1)?))
    }
}

impl<H1, T1, H2, T2> TupleZip<TupleVec<H2, T2>> for TupleVec<H1, T1>
where
    H1: TupleList,
    H2: TupleList,
    T1: TupleZip<T2>,
    T2: TupleList,
{
    type Zipped = TupleVec<(H1, H2), <T1 as TupleZip<T2>>::Zipped>;

    fn zip(self, other: TupleVec<H2, T2>) -> Option<Self::Zipped> {
        if self.0.len() == other.0.len() {
            let new_vec: Vec<(H1, H2)> = self.0.into_iter().zip(other.0.into_iter()).collect();

            Some(TupleVec(new_vec, self.1.zip(other.1)?))
        } else {
            None
        }
    }
}

pub trait TupleForEachFn<A>: HasErr {
    fn apply(&mut self, arg: A) -> Result<(), Self::Err>;
}

pub trait TupleForEach<F: HasErr>: TupleList {
    fn for_each<'a>(self, func: &mut F) -> Result<(), F::Err>;
}

impl<F: HasErr> TupleForEach<F> for () {
    fn for_each(self, _func: &mut F) -> Result<(), F::Err> {
        Ok(())
    }
}

impl<F: TupleForEachFn<H>, H, T: TupleForEach<F>> TupleForEach<F> for (H, T) {
    fn for_each(self, func: &mut F) -> Result<(), F::Err> {
        func.apply(self.0)?;
        self.1.for_each(func)
    }
}

impl<F: HasErr, H: TupleForEach<F>, T: TupleForEach<F>> TupleForEach<F> for TupleVec<H, T> {
    fn for_each(self, func: &mut F) -> Result<(), F::Err> {
        for x in self.0 {
            x.for_each(func)?;
        }
        self.1.for_each(func)
    }
}

pub trait TupleMapFn<A>: HasErr {
    type Output;

    fn apply(&mut self, arg: A) -> Result<Self::Output, Self::Err>;
}

struct CountParams(usize);

impl HasErr for CountParams {
    type Err = std::string::String;
}

impl<S: Shape, E: Dtype, D: DeviceStorage> TupleForEachFn<&Tensor<S, E, D>> for CountParams {
    fn apply(&mut self, arg: &Tensor<S, E, D>) -> Result<(), alloc::string::String> {
        self.0 += arg.shape().num_elements();
        Ok(())
    }
}

pub trait GetTensors {
    type Names: TupleList;
    type Tensors<'a>: TupleList where Self: 'a;
    type TensorsMut<'a>: TupleList where Self: 'a;

    fn get_names(&self, prefix: &str) -> Self::Names;
    fn get_tensors<'a>(&'a self) -> Self::Tensors<'a>;
    fn get_tensors_mut<'a>(&'a mut self) -> Self::TensorsMut<'a>;
}

impl<S: Shape, E: Dtype, D: DeviceStorage> GetTensors for Tensor<S, E, D> {
    type Names = (std::string::String, ());
    type Tensors<'a> = (&'a Self, ());
    type TensorsMut<'a> = (&'a mut Self, ());

    fn get_names(&self, prefix: &str) -> Self::Names {
        use std::string::ToString;

        (prefix.to_string(), ())
    }

    fn get_tensors<'a>(&'a self) -> Self::Tensors<'a> {
        (self, ())
    }

    fn get_tensors_mut<'a>(&'a mut self) -> Self::TensorsMut<'a> {
        (self, ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{modules::Linear, BuildModule},
        tests::*,
    };

    #[test]
    fn test_count_linear_params() {
        let dev: TestDevice = Default::default();
        let linear1: Linear<2, 3, _, _> = BuildModule::build(&dev);
        let mut linear2: Linear<3, 5, _, _> = BuildModule::build(&dev);
        let mut param_count = CountParams(0);

        let tensors = linear1.get_tensors();
        tensors.for_each(&mut param_count).unwrap();

        let tensors1 = linear1.get_tensors();
        let tensors2 = linear2.get_tensors_mut();
        let _zipped = tensors1.zip(tensors2).unwrap();

        let tensors1 = linear1.get_tensors();
        let tensors2 = linear2.get_tensors();
        let _appended: AppendTensors<Linear<2, 3, f32, TestDevice>, Linear<3, 5, f32, TestDevice>> =
            tensors1.append(tensors2);

        assert_eq!(param_count.0, 9);
    }
}
