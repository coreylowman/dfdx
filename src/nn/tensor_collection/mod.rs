//! Traits to define a [TensorCollection] and how to iterate them using [ModuleVisitor].
//! Use [RecursiveWalker] to do the iteration.

mod collection;
mod visitor;

pub use collection::{
    ModuleVisitor, ModuleVisitorOutput, TensorCollection, TensorVisitorOutput, TensorOptions,
};
pub use visitor::{
    RecursiveWalker, TensorVisitor, TensorViewer, ViewTensorMut, ViewTensorName,
    ViewTensorRef,
};
