//! Traits to define a [TensorCollection] and how to iterate them using [ModuleVisitor].
//! Use [RecursiveWalker] to do the iteration.

mod collection;
mod visitor;

pub use collection::{ModuleVisitor, ModuleVisitorOutput, TensorCollection, TensorOptions};
pub use visitor::{
    RecursiveWalker, TensorViewer, TensorVisitor, ViewTensorMut, ViewTensorName, ViewTensorRef,
};
