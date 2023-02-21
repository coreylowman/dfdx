use crate::prelude::*;
use crate::shapes::Dtype;

use std::{fmt::Debug, string::String};

/// A struct which can be used to iterate through all tensors in a module, or to iterate through
/// corresponding tensors in multiple modules of the same type.
///
/// N specifies the number of immutable references to modules of type T this stores, while M
/// specifies the number of mutable references.
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
    /// Creates a new TensorVisitor
    ///
    /// Parameters:
    /// * refs: An array of immutable references to the module type to visit.
    /// * refs_mut: An array of mutable references to the module type to visit.
    /// * name: An optional prefix to each tensor's name. Passing `None` will prevent names from
    /// being tracked, and will provide `None` to [TensorFunction::call].
    /// * func: The [TensorFunction] that this tensor should call on each group of corresponding
    /// tnesors in `refs` and `refs_mut`
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

    /// Calls the stored [TensorFunction] on all groups of corresponding tensors in the modules
    /// stored in `refs` and `refs_mut`.
    ///
    /// For example, if refs consists of two references to [Linear] layers, `[&linear1, &linear2]`,
    /// then the function will be given `[&linear1.weight, &linesr2.weight]` and `[&linear1.bias, &linesr2.bias]`.
    pub fn visit<E: Dtype, D: DeviceStorage>(self) -> Result<(), F::Err>
    where
        F: TensorFunction<N, M, E, D>,
        T: VisitTensors<E, D>,
    {
        VisitTensors::visit_groups(self)
    }

    /// Creates a new TensorVisitor that holds references to a particular field of each module in
    /// refs and refs_mut and calls [TensorVisitor::visit] on it.
    ///
    /// Parameters:
    /// * imm_func: Given a reference to a module, returns an immutable reference to the
    /// field to visit.
    /// * mut_func: Given a reference to a module, returns a mutable reference to the
    /// field to visit.
    /// * name: Specifies the name of the field. This should have a dot at the end unless the
    /// field is a tensor.
    pub fn visit_field<
        E: Dtype,
        D: DeviceStorage,
        T2: Debug,
        F1: FnMut(&T) -> &T2,
        F2: FnMut(&mut T) -> &mut T2,
    >(
        &mut self,
        imm_func: F1,
        mut_func: F2,
        name: &str,
    ) -> Result<(), F::Err>
    where
        F: TensorFunction<N, M, E, D>,
        T2: VisitTensors<E, D>,
    {
        self.map(imm_func, mut_func, name).visit()
    }

    /// Same as [TensorVisitor::visit_field], but appends the options in `options` to each
    /// [`&[TensorFunctionOption]`](TensorFunctionOption) passed into [TensorFunction::call].
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
        T2: VisitTensors<E, D>,
    {
        let mut field_visitor = self.map(func1, func2, name);

        let mut field_options = field_visitor.options.to_vec();
        field_options.extend_from_slice(options);
        field_visitor.options = &field_options;

        field_visitor.visit()
    }
}

/// Configures the behavior of certain [TensorFunction]s.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum TensorFunctionOption {
    /// Prevents tensors from being updated with [GradientUpdate].
    DisableGradientUpdate,
    /// Tells ResetParams to initialize each tensor with a uniform random distribution,
    /// with the first parameter as the lower bound and the second parameter as the upper bound.
    ResetParamsUniform(f64, f64),
    /// Tells ResetParams to initialize each tensor with ones.
    ResetParamsOnes,
}

/// A data type that can be called on a group of tensors, as provided by a [TensorVisitor]
pub trait TensorFunction<const N: usize, const M: usize, E: Dtype, D: DeviceStorage> {
    type Err: std::fmt::Display + Debug;

    /// Updates the  
    fn call<S: Shape>(
        &mut self,
        refs: [&Tensor<S, E, D>; N],
        refs_mut: [&mut Tensor<S, E, D>; M],
        name: Option<String>,
        options: &[TensorFunctionOption],
    ) -> Result<(), Self::Err>;
}

/// Specifies how a [TensorVisitor] should traverse the fields of a Module.
///
/// Implementing this automatically implements [ResetParams], [GradientUpdate], [SaveToNpz],
/// [LoadFromNpz], and [CountParams].
pub trait VisitTensors<E: Dtype, D: DeviceStorage>:
    Sized + Debug
{
    fn visit_groups<const N: usize, const M: usize, F: TensorFunction<N, M, E, D>>(
        visitor: TensorVisitor<N, M, Self, F>,
    ) -> Result<(), F::Err>;

    /// Calls `func` on immutable references to all tensors in `self`.
    fn visit<F: TensorFunction<1, 0, E, D>>(&self, func: &mut F) -> Result<(), F::Err> {
        TensorVisitor::new([self], [], None, func).visit()
    }

    /// Calls `func` on immutable references to all tensors in `self`, passing each tensor's name
    /// prefixed by the name parameter into each call of [TensorFunction::call]
    fn visit_with_name<F: TensorFunction<1, 0, E, D>>(
        &self,
        name: String,
        func: &mut F,
    ) -> Result<(), F::Err> {
        TensorVisitor::new([self], [], Some(name), func).visit()
    }

    /// Call `func` on mutable references to all tensors in `self`.
    fn visit_mut<F: TensorFunction<0, 1, E, D>>(&mut self, func: &mut F) -> Result<(), F::Err> {
        VisitTensors::visit_groups(TensorVisitor::new([], [self], None, func))
    }

    /// Call `func` on mutable references to all tensors in `self`, passing each tensor's name
    /// prefixed by the name parameter into each call of [TensorFunction.call]
    fn visit_mut_with_name<F: TensorFunction<0, 1, E, D>>(
        &mut self,
        name: String,
        func: &mut F,
    ) -> Result<(), F::Err> {
        TensorVisitor::new([], [self], Some(name), func).visit()
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage>
    VisitTensors<E, D> for Tensor<S, E, D>
{
    fn visit_groups<const N: usize, const M: usize, F: TensorFunction<N, M, E, D>>(
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
