use crate::shapes::Dtype;

use crate::prelude::*;
use std::{
    fmt::Debug,
    string::{String, ToString},
};

pub struct TensorVisitor<'a, const N: usize, const M: usize, T, F> {
    refs: [&'a T; N],
    refs_mut: [&'a mut T; M],
    name: Option<String>,
    func: &'a mut F,
    options: &'a [TensorFunctionOption],
}

// TODO? : Prevent heap allocation using unsafe or ArrayVec
fn map_mut<'a, A: 'a + Debug, B: 'a + Debug, const N: usize, F: FnMut(&'a mut A) -> &'a mut B>(
    arr: &'a mut [A; N],
    func: F,
) -> [&'a mut B; N] {
    let vec: std::vec::Vec<_> = arr.iter_mut().map(func).collect();

    vec.try_into().unwrap()
}

impl<'a, const N: usize, const M: usize, T: Debug, F> TensorVisitor<'a, N, M, T, F> {
    pub fn new(
        refs: [&'a T; N],
        refs_mut: [&'a mut T; M],
        name: Option<String>,
        func: &'a mut F,
    ) -> Self {
        Self {
            refs,
            refs_mut,
            name,
            func,
            options: &[],
        }
    }

    fn map<T2: Debug, F1: FnMut(&T) -> &T2, F2: FnMut(&mut T) -> &mut T2>(
        &mut self,
        func1: F1,
        mut func2: F2,
        name: &str,
    ) -> TensorVisitor<N, M, T2, F> {
        TensorVisitor {
            refs: self.refs.map(func1),
            refs_mut: map_mut(&mut self.refs_mut, |x| func2(*x)),
            name: self
                .name
                .as_ref()
                .map(|prefix| std::format!("{prefix}{name}")),
            func: self.func,
            options: self.options,
        }
    }

    pub fn visit<E: Dtype, D: DeviceStorage>(self) -> Result<(), F::Err>
    where
        F: TensorFunction<N, M, E, D>,
        T: VisitTensorGroups<N, M, E, D>,
    {
        VisitTensorGroups::visit_groups(self)
    }

    pub fn visit_field<
        E: Dtype,
        D: DeviceStorage,
        T2: Debug,
        F1: FnMut(&T) -> &T2,
        F2: FnMut(&mut T) -> &mut T2,
    >(
        &mut self,
        func1: F1,
        func2: F2,
        name: &str,
    ) -> Result<(), F::Err>
    where
        F: TensorFunction<N, M, E, D>,
        T2: VisitTensorGroups<N, M, E, D>,
    {
        self.map(func1, func2, name).visit()
    }

    pub fn visit_field_with_options<
        E: Dtype,
        D: DeviceStorage,
        T2: Debug,
        F1: FnMut(&T) -> &T2,
        F2: FnMut(&mut T) -> &mut T2,
    >(
        &mut self,
        func1: F1,
        func2: F2,
        name: &str,
        options: &[TensorFunctionOption],
    ) -> Result<(), F::Err>
    where
        F: TensorFunction<N, M, E, D>,
        T2: VisitTensorGroups<N, M, E, D>,
    {
        let mut field_visitor = self.map(func1, func2, name);

        let mut field_options = field_visitor.options.to_vec();
        field_options.extend_from_slice(options);
        field_visitor.options = &field_options;

        field_visitor.visit()
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum TensorFunctionOption {
    DisableGradientUpdate,
    ResetParamsDistribution(f64, f64),
    ResetParamsOnes,
}

pub trait TensorFunction<const N: usize, const M: usize, E: Dtype, D: DeviceStorage> {
    type Err: std::fmt::Display + Debug;

    fn call<S: Shape>(
        &mut self,
        refs: [&Tensor<S, E, D>; N],
        refs_mut: [&mut Tensor<S, E, D>; M],
        name: Option<String>,
        options: &[TensorFunctionOption],
    ) -> Result<(), Self::Err>;
}

pub trait VisitTensorGroups<const N: usize, const M: usize, E: Dtype, D: DeviceStorage>:
    Sized
{
    fn visit_groups<F: TensorFunction<N, M, E, D>>(
        visitor: TensorVisitor<N, M, Self, F>,
    ) -> Result<(), F::Err>;
}

impl<const N: usize, const M: usize, S: Shape, E: Dtype, D: DeviceStorage>
    VisitTensorGroups<N, M, E, D> for Tensor<S, E, D>
{
    fn visit_groups<F: TensorFunction<N, M, E, D>>(
        visitor: TensorVisitor<N, M, Self, F>,
    ) -> Result<(), F::Err> {
        visitor.func.call(
            visitor.refs,
            visitor.refs_mut,
            visitor.name,
            visitor.options,
        )
    }
}

pub trait VisitTensors<E: Dtype, D: DeviceStorage>: VisitTensorGroups<1, 0, E, D> + Debug {
    fn visit<F: TensorFunction<1, 0, E, D>>(&self, func: &mut F) -> Result<(), F::Err> {
        TensorVisitor::new([self], [], None, func).visit()
    }

    fn visit_with_name<F: TensorFunction<1, 0, E, D>>(
        &self,
        name: &str,
        func: &mut F,
    ) -> Result<(), F::Err> {
        TensorVisitor::new([self], [], Some(name.to_string()), func).visit()
    }
}

impl<E: Dtype, D: DeviceStorage, T> VisitTensors<E, D> for T where
    T: VisitTensorGroups<1, 0, E, D> + Debug
{
}

pub trait VisitTensorsMut<E: Dtype, D: DeviceStorage>:
    VisitTensorGroups<0, 1, E, D> + Debug
{
    fn visit_mut<F: TensorFunction<0, 1, E, D>>(&mut self, func: &mut F) -> Result<(), F::Err> {
        VisitTensorGroups::visit_groups(TensorVisitor::new([], [self], None, func))
    }

    fn visit_mut_with_name<F: TensorFunction<0, 1, E, D>>(
        &mut self,
        name: &str,
        func: &mut F,
    ) -> Result<(), F::Err> {
        TensorVisitor::new([], [self], Some(name.to_string()), func).visit()
    }
}

impl<E: Dtype, D: DeviceStorage, T> VisitTensorsMut<E, D> for T where
    T: VisitTensorGroups<0, 1, E, D> + Debug
{
}

struct CountParamsVisitor(usize);

impl<E: Dtype, D: DeviceStorage> TensorFunction<1, 0, E, D> for CountParamsVisitor {
    type Err = String;

    fn call<S: Shape>(
        &mut self,
        refs: [&Tensor<S, E, D>; 1],
        _refs_mut: [&mut Tensor<S, E, D>; 0],
        _name: Option<String>,
        _options: &[TensorFunctionOption],
    ) -> Result<(), Self::Err> {
        self.0 += refs[0].shape().num_elements();
        Ok(())
    }
}

pub trait CountParams<E: Dtype, D: DeviceStorage>: VisitTensors<E, D> {
    fn try_param_count(&self) -> Result<usize, String> {
        let mut visitor = CountParamsVisitor(0);
        self.visit(&mut visitor)?;
        Ok(visitor.0)
    }

    fn param_count(&self) -> usize {
        self.try_param_count().unwrap()
    }
}

impl<E: Dtype, D: DeviceStorage, T: VisitTensors<E, D>> CountParams<E, D> for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_linear_count_params() {
        let dev: TestDevice = Default::default();
        let linear: crate::nn::modules::Linear<2, 3, TestDtype, _> = BuildModule::build(&dev);

        assert_eq!(linear.param_count(), 9);
    }
}
