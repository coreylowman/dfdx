//! Traits to define a [TensorCollection] and how to iterate them using [ModuleVisitor].
//! Use [RecursiveWalker] to do the iteration and [TensorVisitor] to define functions to iterate
//! through and/or construct Modules.

mod collection;
mod visitor;
mod visitor_impls;

pub use collection::{HyperparameterOptions, ModuleVisitor, TensorCollection, TensorOptions};
pub use visitor::{
    HyperparameterField, ModuleField, ModuleFields, RecursiveWalker, TensorField, TensorViewer,
    TensorVisitor, ViewTensorMut, ViewTensorName, ViewTensorRef,
};
