//! Traits to define a [TensorCollection] and how to iterate them using [ModuleVisitor].
//! Use [RecursiveWalker] to do the iteration.

mod collection;
mod visitor;
mod visitor_impls;

pub use collection::{ModuleVisitor, TensorCollection, TensorOptions};
pub use visitor::{
    ModuleField, ModuleFields, RecursiveWalker, TensorField, TensorViewer, TensorVisitor,
    ViewTensorMut, ViewTensorName, ViewTensorRef,
};
