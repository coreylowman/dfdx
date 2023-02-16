use crate::shapes::Dtype;

use crate::prelude::*;
use std::{fmt::Debug, string::String};

pub struct ModuleGroup<'a, const N: usize, const M: usize, T> {
    pub refs: [&'a T; N],
    pub refs_mut: [&'a mut T; M],
    pub name: Option<String>,
}

// TODO? : Prevent heap allocation using unsafe or ArrayVec
fn map_mut<'a, A: 'a + Debug, B: 'a + Debug, const N: usize, F: FnMut(&'a mut A) -> &'a mut B>(
    arr: &'a mut [A; N],
    func: F,
) -> [&'a mut B; N] {
    let vec: std::vec::Vec<_> = arr.iter_mut().map(func).collect();

    vec.try_into().unwrap()
}

impl<'a, const N: usize, const M: usize, T: Debug> ModuleGroup<'a, N, M, T> {
    pub fn new(refs: [&'a T; N], refs_mut: [&'a mut T; M], name: Option<String>) -> Self {
        Self {
            refs,
            refs_mut,
            name,
        }
    }

    pub fn map<T2: Debug, F1: FnMut(&T) -> &T2, F2: FnMut(&mut T) -> &mut T2>(
        &mut self,
        func1: F1,
        mut func2: F2,
        name: &str,
    ) -> ModuleGroup<N, M, T2> {
        ModuleGroup {
            refs: self.refs.map(func1),
            refs_mut: map_mut(&mut self.refs_mut, |x| func2(*x)),
            name: self
                .name
                .as_ref()
                .map(|prefix| std::format!("{prefix}{name}")),
        }
    }

    pub fn visit<E: Dtype, D: DeviceStorage, F: TensorVisitor<N, M, E, D>>(
        self,
        func: &mut F,
    ) -> Result<(), D::Err>
    where
        T: VisitTensorGroups<N, M, E, D>,
    {
        VisitTensorGroups::visit_groups(self, func)
    }
}

#[non_exhaustive]
pub enum TensorVisitorOption {
    DoGradientUpdate(bool),
}

pub trait TensorVisitor<const N: usize, const M: usize, E: Dtype, D: DeviceStorage> {
    fn call<S: Shape>(&mut self, tensors: ModuleGroup<N, M, Tensor<S, E, D>>)
        -> Result<(), D::Err>;
    fn set_option(&mut self, _option: TensorVisitorOption) {}
}

pub trait VisitTensorGroups<const N: usize, const M: usize, E: Dtype, D: DeviceStorage>:
    Sized
{
    fn visit_groups<F: TensorVisitor<N, M, E, D>>(
        self_refs: ModuleGroup<N, M, Self>,
        func: &mut F,
    ) -> Result<(), D::Err>;
}

impl<const N: usize, const M: usize, S: Shape, E: Dtype, D: DeviceStorage>
    VisitTensorGroups<N, M, E, D> for Tensor<S, E, D>
{
    fn visit_groups<F: TensorVisitor<N, M, E, D>>(
        self_refs: ModuleGroup<N, M, Self>,
        func: &mut F,
    ) -> Result<(), D::Err> {
        func.call(self_refs)
    }
}

pub trait VisitTensors<E: Dtype, D: DeviceStorage>: VisitTensorGroups<1, 0, E, D> + Debug {
    fn visit<F: TensorVisitor<1, 0, E, D>>(&self, func: &mut F) -> Result<(), D::Err> {
        VisitTensorGroups::visit_groups(ModuleGroup::new([self], [], None), func)
    }
}

impl<E: Dtype, D: DeviceStorage, T> VisitTensors<E, D> for T where
    T: VisitTensorGroups<1, 0, E, D> + Debug
{
}

pub trait VisitTensorsMut<E: Dtype, D: DeviceStorage>:
    VisitTensorGroups<0, 1, E, D> + Debug
{
    fn visit_mut<F: TensorVisitor<0, 1, E, D>>(&mut self, func: &mut F) -> Result<(), D::Err> {
        VisitTensorGroups::visit_groups(ModuleGroup::new([], [self], None), func)
    }
}

impl<E: Dtype, D: DeviceStorage, T> VisitTensorsMut<E, D> for T where
    T: VisitTensorGroups<0, 1, E, D> + Debug
{
}

struct CountParamsVisitor(usize);

impl<E: Dtype, D: DeviceStorage> TensorVisitor<1, 0, E, D> for CountParamsVisitor {
    fn call<S: Shape>(
        &mut self,
        tensors: ModuleGroup<1, 0, Tensor<S, E, D>>,
    ) -> Result<(), D::Err> {
        self.0 += tensors.refs[0].shape().num_elements();
        Ok(())
    }
}

pub trait CountParams<E: Dtype, D: DeviceStorage>: VisitTensors<E, D> {
    fn try_param_count(&self) -> Result<usize, D::Err> {
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
