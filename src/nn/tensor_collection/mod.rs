//! Traits to define a [TensorCollection] and how to iterate them using [ModuleVisitor].
//! Use [RecursiveWalker] to do the iteration.

mod collection;
mod visitor;

pub use collection::{
    ModuleVisitor, ModuleVisitorOutput, TensorCollection, TensorFunctionOutput, TensorOptions,
};
pub use visitor::{
    RecursiveWalker, TensorFunction, TensorViewer, TensorVisitor, ViewTensorMut, ViewTensorName,
    ViewTensorRef,
};
